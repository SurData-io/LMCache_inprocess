# LMCache InProcess

## 2026-11-10 没有 preloaded 且 HBM 有空就报 miss

调度器侧 `TierAdmission` 相位是 `ready` 时，`get_num_new_matched_tokens` 仍返回 host 里已经有的 token 数，vLLM 按原路径把这段 KV 拷进 HBM。这一轮没有任何 `ready`，并且 vLLM 0.33 的 `scheduler.kv_cache_manager.block_pool.get_num_free_blocks()` 大于 0 时，`absent`、`pending`、`queued` 返回 `0`，GPU 重算。已经有别的 `ready`，或者读不到空闲块、空闲块是 0，这三种相位仍返回 `None`。`pending` 和 `queued` 被报成 miss 时调用已有的 `cancel_lookup`，迟到的 ACK 不能把相位改回去。

### 已完成
- `admission_tokens` 增加 `any_ready` 和 `hbm_free`。`TierAdmission.has_ready` 只在锁里看相位。`scheduler_hbm_has_free_block` 只认上面这个 v0.33.0 的取数函数，属性缺失或调用失败当成没有空块。
- `on_vllm_schedule` 每轮读一次空闲块，记在 `_tier_hbm_free`。第一轮调度之前这个值是假。
- 本机没有 `pytest` 和 `lmcache_native`。用导入替身跑了 `tests/inprocess/v1/test_tier_pipeline.py`：34 项里 17 项因缺少 `lmcache.lmcache_native` 跳过，其余 17 项通过，其中包括新的空 HBM 报 miss 和空闲块读取。没有跑 800×2。host 池大小没有改。

### 尚未完成
- 没有在 GPU 上复测 800×2，所以还不知道第一轮那 21.9 秒有没有缩短。
- 镜像里没有装这个源。要进下一轮 bench 得先打进 `vllm-openai:v0.33.0-lmcache0.5.7-pr9` 那条镜像。



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
