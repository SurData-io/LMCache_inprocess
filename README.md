# LMCache InProcess

## 2026-11-12 按可分配块数回收 store 池

`store()` 逐块申请。一块是对齐后的 chunk。几段各自装得下一块的洞，块数相加。小于一块的碎片算 0。`ensure_free_chunks` 在可分配块数少于这一拍的预算时继续驱逐 store 池里 `can_evict` 的对象，不再因为「原始空闲字节已经够、最长空段至少一块」就停。返回的是可分配块数。`wait_for_save` 用这个数扣每一条请求。在飞写和 pin 住的对象仍然不回收。32 块的步上限、8GiB / 12GiB 两池不变。

`20261112-134737` 里有 50 步一进来最长空段是 7，2048 token 要 8 块，后面的请求被整条跳过，能驱逐的 KV 还留在池里。前 50 秒 `Stored` 约 230–330MiB/s，和每步 32 块的上限一致，不是盘写不及 3090。

### 已完成
- 水位按可分配块数继续驱逐，返回这个块数。
- 入场按可分配块数扣减。一段 7 块不够 8 时会再腾。
- 隔块的洞可以相加后做 8 次逐块申请。引用大于 1 和 preload 池的对象仍不回收。

### 尚未完成
- 没有把这次改动打进 GPU 镜像，也没有复测 800×2。

## 2026-11-12 store 池和 preload 池

`max_local_cpu_size` 正好是 20 时，默认的非分页 `MixedMemoryAllocator` 拆成两块。store 池 `8589934592`（8GiB），preload 池 `12884901888`（12GiB）。这两个数和 SurData 直连的 `STORE_POOL_DEFAULT` / `PRELOAD_POOL_DEFAULT` 相同，不从 `MemTotal`、`max_local_cpu_size` 或 OOM 日志推。没有新配置项。P2P、NIXL、io_uring，以及单测注入的小分配器，仍是一个分配器。只有一个分配器时 `purpose` 不起作用。

D2H 的 `store` / `store_layer` 用 `purpose="store"`。回收只拿 store 池里 `can_evict` 的热缓存对象：没被 pin，且引用等于 1。正在写盘的对象引用大于 1，留在 store 池里。入场看的是能逐块申请的块数，见上一节。装不下的整条请求跳过，不调用 `batched_from_gpu`。`drop_on_alloc_miss` 时分配仍然 `eviction=False`。

读盘三处用 `purpose="preload"`：`batched_get_non_blocking`、`batched_get_blocking`、`load_bytes_from_disk`。`busy_loop` 保持原来的值。preload 池分不出连续的一块时停在当前 key，不进 store 池去抢。

### 已完成
- 20GiB 默认分配器拆成 8GiB store 和 12GiB preload。
- 当时水位在字节够且最长空段至少一块时停，入场看最长空段。停止条件已改成可分配块数，见上一节。
- 读盘只从 preload 池申请。
- `tests/inprocess/v1/test_local_cpu_pre_evict.py` 覆盖两池隔离、在飞写不回收、隔块洞相加、七块时继续驱逐。

### 尚未完成
- 两池拆分已打进镜像 `vllm-openai:v0.33.0-lmcache0.5.7-pr9-767b77b`，并跑过 `20261112-134737`。按块数回收还没有打进镜像。

## 2026-11-12 写盘直接进线程池

`submit_put_task` 不再把落盘协程交给 `storage-manager-event-loop`。`LocalDiskWorker.submit_put` 把 `async_save_bytes_to_disk` 直接提交给名为 `disk-put` 的 `ThreadPoolExecutor`，大小仍是已有的 `disk_io_threads`（默认 4）。预取还走原来的优先级队列。没有新配置项。

`put_tasks` 从列表改成集合。登记、查找、删除都是 O(1)，不再在 `disk_lock` 里线性扫描在飞 key。`write_file` 抛错时减引用、从在飞集合去掉 key、退回这次加上的用量和 `current_cache_size`，不 `insert_key`。提交进线程池失败时同样减引用、去掉 key，并退回已经计入的 `current_cache_size`。

### 已完成
- 写盘直接进线程池。在飞 key 用集合。写失败释放引用和配额。
- `tests/inprocess/v1/storage_backend/test_local_disk_put_admission.py` 覆盖 4 路同时停在 `write_file`，以及写失败后引用下降、key 不进 `dict`。

### 尚未完成
- 没有把这次改动打进 GPU 镜像，也没有复测 800×2。

## 2026-11-11 对齐预算，腾不够就丢掉 KV

预算和空闲槽都按对齐后的整块算。`aligned_savable_tokens` 与 `wait_for_save` 的实际写入对齐：`slot_mapping` 和 token 长度不一致算 0；`skip_leading_tokens` 向下对齐到 `chunk_size`；不是最后一段 prefill 且没开 blending 时，结尾也向下对齐。块数是 `savable // chunk_size`，尾巴不再 `ceil`。layerwise 一条 chunk 要 `num_layers` 个对象，槽位按这个倍数算，token 块数仍钳到 32。

空闲槽用 `get_full_chunk_size_bytes()`（没有 metadata 时用已在池里的对象物理大小）再按 allocator 的 `align_bytes` 向上对齐。地址空闲链表里每个连续块贡献 `size // aligned`，小于一块的碎片算 0。`ensure_free_chunks` 只驱逐差额，一次 `batched_remove(force=False)`，pin 住的和在飞落盘不动，也不 sleep。返回值是驱逐之后的对齐空闲槽数。看不到空闲链表时返回 0。

`wait_for_save` 先对整批 `lookup_unpin`，再预驱逐，再按请求扣槽。一条请求的块数大于剩余槽就打日志并跳过 `store`，不做 D2H。`save_kv_layer` 在第一次建 storer 之前同样先 unpin 再算槽，不够就不建 `store_layer`。`kv_consumer` 只 unpin、不存。这条路径的 `store` / `store_layer` 带 `drop_on_alloc_miss`：分配失败时把已经拿到的对象 `ref_count_down`，在 `batched_from_gpu` 之前返回，不再「失败一次、驱逐一个、再分配」。没有新配置项，`reserve_local_cpu_size` 和 SurData 直连池没有改。

### 已完成
- 预算按 chunk 对齐整除。空闲槽按对齐后的连续空闲块计数。腾不够就丢这条请求的 KV，不做 D2H。
- 本机用 venv 的 torch 跑了 `tests/inprocess/v1/test_local_cpu_pre_evict.py`：7 项通过。覆盖对齐块数、碎片算 0、腾不出预算时返回的空闲槽小于预算、看不到空闲链表时返回 0。没有跑 800×2。

### 尚未完成
- 没有把这次改动打进 GPU 镜像，也没有复测 800×2。

## 2026-11-11 batch 组建后预驱逐 20GiB local CPU 池

`store` 逐 chunk 调用 `allocate`。20GiB pinned 池写满后，每次分配先失败再从 hot cache 驱逐 1 个，这段耗时记在 `offload_time` 里。`20261110-222612` 的后半段 offload 中位从 9.7 ms 升到 52.6 ms。落盘的 `put_time` 中位仍是 0.56 ms。

上一节把「不够就退回逐 chunk 驱逐」改成了「腾不够就丢掉这条 KV」。这里留下当时的做法，方便对照。

### 已完成
- `chunks_for_save_batch`、`ensure_free_chunks`、`LMCacheEngine.pre_evict_for_save`。
- `wait_for_save` 在非 layerwise 的 store 循环前调用一次。`save_kv_layer` 只在还没建过 layerwise storer 时调用一次。`kv_consumer` 和 engine 未初始化的分支不调用。

### 尚未完成
- 没有把预驱逐打进 GPU 镜像，也没有复测 800×2。`get_size_bytes` 改成 allocator 初始化时再导入，避免 store 预驱逐这条导入链去拉 vLLM KV format 注册表。

## 2026-11-10 没有 preloaded 且 HBM 有空就报 miss

调度器侧 `TierAdmission` 相位是 `ready` 时，`get_num_new_matched_tokens` 仍返回 host 里已经有的 token 数，vLLM 按原路径把这段 KV 拷进 HBM。这一轮没有任何 `ready`，并且 vLLM 0.33 的 `scheduler.kv_cache_manager.block_pool.get_num_free_blocks()` 大于 0 时，`absent`、`pending`、`queued` 返回 `0`，GPU 重算。已经有别的 `ready`，或者读不到空闲块、空闲块是 0，这三种相位仍返回 `None`。`pending` 和 `queued` 被报成 miss 时调用已有的 `cancel_lookup`，迟到的 ACK 不能把相位改回去。

### 已完成
- `admission_tokens` 增加 `any_ready` 和 `hbm_free`。`TierAdmission.has_ready` 只在锁里看相位。`scheduler_hbm_has_free_block` 只认上面这个 v0.33.0 的取数函数，属性缺失或调用失败当成没有空块。
- `on_vllm_schedule` 每轮读一次空闲块，记在 `_tier_hbm_free`。第一轮调度之前这个值是假。
- 本机没有 `pytest` 和 `lmcache_native`。用导入替身跑了 `tests/inprocess/v1/test_tier_pipeline.py`：34 项里 17 项因缺少 `lmcache.lmcache_native` 跳过，其余 17 项通过，其中包括新的空 HBM 报 miss 和空闲块读取。没有跑 800×2。host 池大小没有改。

### 尚未完成
- 2026-11-10 22:26 复测，戳 `20261110-222612`。镜像 `vllm-openai:v0.33.0-lmcache0.5.7-pr9-3c3398a`，两个开关都开，盘是 `/mnt/xfs` 上的 XFS（`/mnt/xfs/surdata-xfs/native/kvcache`）。综合 408.5 tok/s = (51170+35114)/211.217。Turn1 163.82s，4.88 req/s，312.36 tok/s，51170 token，TTFT 均/中/P99 60744.88/53040.20/140117.97 ms。Turn2 47.40s，16.88 req/s，740.85 tok/s，35114 token，TTFT 96373.70/93858.72/129239.64 ms。`wall_s=211.217`，重叠 196.501s。两轮都是 800 成功、0 失败。`EXIT_RC=0`。比 `64a2a5d` 的 566.5 和原生 XFS 的 656.1 都低。没有推送。



`lmcache_inprocess` packages LMCache's legacy in-process engine separately from
the main [LMCache](https://github.com/LMCache/LMCache) distribution. It keeps the
existing Python import path, `lmcache.inprocess`, so current vLLM, TensorRT-LLM,
configuration, and plugin integrations do not need an API rename.

The wheel also supplies
`lmcache.integration.vllm.vllm_v1_adapter` as a compatibility re-export because
vLLM's connector imports that canonical path. The implementation remains under
`lmcache.inprocess.integration.vllm`.

The source snapshot is synchronized through LMCache `dev` commit
[`05fdc2013b41484f0d834ec2d30b019fd426a56c`](https://github.com/LMCache/LMCache/commit/05fdc2013b41484f0d834ec2d30b019fd426a56c).
The exact source revision is also recorded in [`UPSTREAM_REVISION`](UPSTREAM_REVISION).

## Install

Install a compatible LMCache build first, then this package:

```bash
python -m pip install 'lmcache>=0.5.7rc3'
python -m pip install lmcache_inprocess
```

For development against LMCache `dev`:

```bash
NO_GPU_EXT=1 python -m pip install \
  'lmcache @ git+https://github.com/LMCache/LMCache.git@dev'
python -m pip install -e . --no-deps
```

The distribution is pure Python. Native extensions remain owned and shipped by
the `lmcache` dependency. A CLI-only `NO_NATIVE_EXT=1` LMCache installation is
not sufficient because the in-process engine requires LMCache's common native
operations; CPU validation uses `NO_GPU_EXT=1`.

## Compatibility contract

- Distribution name: `lmcache_inprocess`
- Import namespace: `lmcache.inprocess`
- Base dependency: `lmcache>=0.5.7rc3`
- Python: 3.10 through 3.14
- Supported source of shared native extensions: the installed `lmcache` package

Because both distributions contribute files below the top-level `lmcache`
directory, install and uninstall them with the same environment manager. A
future LMCache release should stop shipping `lmcache/inprocess` before both
packages are published as stable releases, avoiding overlapping wheel ownership.

## Development and validation

```bash
python -m pip install -r requirements/test.txt
python -m build
python -m twine check dist/*
python -m pytest -q tests/smoke
python -m pytest -q tests/inprocess/v1/test_mock_gpu_connector.py
```

CI builds and installs the wheel in a clean environment, runs CPU smoke tests,
and tests compatibility against a pinned LMCache revision and current LMCache
`dev`. GPU end-to-end validation is available through the manual workflow in
`.github/workflows/gpu-e2e.yml` on a repository runner labeled
`self-hosted`, `linux`, `x64`, `gpu`, and `h200`.

## Repository layout

- `lmcache/inprocess/`: package source
- `tests/inprocess/`: copied unit and integration tests
- `examples/inprocess/`: in-process examples
- `benchmarks/inprocess/`: in-process benchmarks
- `docs/source/legacy/`: in-process documentation migrated from LMCache
- `scripts/sync_from_lmcache.sh`: reproducible upstream refresh helper

## Release

Create a GitHub release with a `vX.Y.Z` tag. The release workflow builds the
source distribution and universal wheel, validates both, uploads them as GitHub
release artifacts, and publishes to PyPI through Trusted Publishing when the
`pypi` environment is configured for this repository.
