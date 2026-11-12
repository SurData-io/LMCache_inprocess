# SPDX-License-Identifier: Apache-2.0
"""Hot-cache slots are freed once for a store batch, before per-chunk allocate."""

# Standard
import logging

# Third Party
import torch

# First Party
from lmcache.inprocess.observability import LMCStatsMonitor
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.storage_backend.local_cpu_backend import (
    PRELOAD_POOL_BYTES,
    STORE_POOL_BYTES,
    STORE_PREEVICT_CHUNK_CAP,
    LocalCPUBackend,
    aligned_free_slot_count,
    aligned_savable_tokens,
    chunks_for_save_batch,
    largest_contiguous_slot_count,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator

_SHAPE = torch.Size([2, 16, 8, 128])
_DTYPE = torch.bfloat16


def _key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=_DTYPE,
    )


def _backend(pool_bytes: int) -> tuple[LocalCPUBackend, MixedMemoryAllocator]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="pre_evict_test",
    )
    allocator = MixedMemoryAllocator(pool_bytes)
    backend = LocalCPUBackend(config=config, memory_allocator=allocator)
    return backend, allocator


def _fill(backend: LocalCPUBackend) -> list:
    """Allocate until the pinned pool cannot hold another chunk."""
    resident = []
    while True:
        memory_obj = backend.allocate(_SHAPE, _DTYPE, eviction=False)
        if memory_obj is None:
            break
        key = _key(len(resident))
        backend.submit_put_task(key, memory_obj)
        # submit_put_task takes its own ref. Drop the allocator ref so the
        # hot cache is the only owner and the chunk can be evicted.
        memory_obj.ref_count_down()
        resident.append(key)
    assert resident, "pool accepted no chunks"
    return resident


def _alloc_failures_during(callback) -> int:
    logger = logging.getLogger("lmcache")
    failures = []

    class _Handler(logging.Handler):
        def emit(self, record):
            if "Failed to allocate" in record.getMessage():
                failures.append(record)

    handler = _Handler(level=logging.WARNING)
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        callback()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return len(failures)


class _FreeBlock:
    def __init__(self, size: int) -> None:
        self.size = size


def test_chunks_for_save_batch_uses_the_step_cap():
    """One 2048-token request is 8 chunks. A partial tail is not an extra chunk."""
    assert chunks_for_save_batch([2048], 256) == 8
    assert chunks_for_save_batch([2048, 1024, 0], 256) == 12
    assert chunks_for_save_batch([8192, 8192], 256) == STORE_PREEVICT_CHUNK_CAP
    assert chunks_for_save_batch([2000], 256) == 7
    assert chunks_for_save_batch([100], 0) == 0
    assert chunks_for_save_batch([], 256) == 0


def test_aligned_savable_tokens_drops_partial_tail_and_mismatched_slots():
    """Budget is whole chunks. 2000 tokens is 7 chunks, not ceil to 8."""
    assert (
        aligned_savable_tokens(2000, 0, 256, last_prefill=False, blending=False)
        == 7 * 256
    )
    assert (
        aligned_savable_tokens(2000, 0, 256, last_prefill=True, blending=False)
        == 7 * 256
    )
    # 300 aligns down to 256, leaving 7 chunks of a 2048-token request.
    assert (
        aligned_savable_tokens(2048, 300, 256, last_prefill=True, blending=False)
        == 7 * 256
    )
    assert (
        aligned_savable_tokens(
            2048, 0, 256, last_prefill=True, blending=False, slot_count=100
        )
        == 0
    )


def test_aligned_free_slot_count_ignores_fragments():
    """A hole smaller than one aligned chunk is not a free slot."""
    aligned = 8192
    blocks = [
        _FreeBlock(aligned - 1),
        _FreeBlock(aligned * 2 + 100),
        _FreeBlock(100),
    ]
    assert aligned_free_slot_count(blocks, aligned) == 2
    assert aligned_free_slot_count(blocks, 0) == 0


def test_ensure_free_chunks_on_a_full_pool_then_allocates():
    """A full pool evicts the requested slots in one pass.

    The following allocations must hit those slots. They must not log
    ``Failed to allocate``, which is the per-chunk miss this pass removes.
    """
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 8
        before = len(backend.hot_cache)
        free_slots = backend.ensure_free_chunks(8)
        assert free_slots == 8
        assert len(backend.hot_cache) == before - 8

        allocated = []

        def _allocate_eight():
            for _ in range(8):
                memory_obj = backend.allocate(_SHAPE, _DTYPE, eviction=False)
                assert memory_obj is not None
                allocated.append(memory_obj)

        assert _alloc_failures_during(_allocate_eight) == 0
        for memory_obj in allocated:
            memory_obj.ref_count_down()
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_ensure_free_chunks_leaves_pinned_chunks():
    """Pinned hot-cache entries are not part of the free-slot budget."""
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 4
        pinned = resident[:2]
        for key in pinned:
            assert backend.hot_cache[key].pin()
        free_slots = backend.ensure_free_chunks(len(resident))
        assert free_slots == len(resident) - 2
        assert free_slots < len(resident)
        for key in pinned:
            assert key in backend.hot_cache
            assert backend.hot_cache[key].is_pinned
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_ensure_free_chunks_does_not_evict_when_slots_are_free():
    """Already-free slots cover the batch, so the hot cache stays."""
    backend, allocator = _backend(4 * 1024 * 1024)
    try:
        memory_obj = backend.allocate(_SHAPE, _DTYPE, eviction=False)
        assert memory_obj is not None
        key = _key(0)
        backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()
        free_slots = backend.ensure_free_chunks(8)
        assert free_slots >= 8
        assert key in backend.hot_cache
        assert len(backend.hot_cache) == 1
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_allocatable_slots_add_holes_and_ignore_fragments():
    """Two holes of three chunks are six slots. A fragment smaller than one is zero."""
    aligned = 8192
    blocks = [
        _FreeBlock(aligned * 3),
        _FreeBlock(aligned * 3),
        _FreeBlock(aligned - 1),
    ]
    assert aligned_free_slot_count(blocks, aligned) == 6
    assert largest_contiguous_slot_count(blocks, aligned) == 3
    assert aligned_free_slot_count([_FreeBlock(aligned - 1)], aligned) == 0
    assert aligned_free_slot_count(blocks, 0) == 0


def test_disk_budget_counts_preload_slots_not_store_free_bytes():
    """A disk GET is budgeted from the preload pool's one-chunk holes.

    Filling the store pool does not change that number. A fragment smaller
    than one chunk is not part of it.
    """
    chunk = 65536
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="pre_evict_preload_budget",
    )
    store = MixedMemoryAllocator(1 << 20)
    preload = MixedMemoryAllocator(chunk * 2 + 4096)
    backend = LocalCPUBackend(config=config, memory_allocator=store)
    backend.preload_allocator = preload
    backend.store_allocator = store
    backend._split_pools = True
    backend._full_chunk_size_bytes = chunk
    try:
        before = backend.host_allocatable_bytes(chunk)
        stats = backend._pool_free_stats(preload, chunk)
        assert stats is not None and before is not None
        assert stats[2] == 2
        assert before == 2 * chunk
        assert stats[0] > before
        store_obj = backend.allocate(_SHAPE, _DTYPE, eviction=False, purpose="store")
        assert store_obj is not None
        assert backend.host_allocatable_bytes(chunk) == before
        store_obj.ref_count_down()

        import sys

        pkg = sys.modules.get("lmcache.inprocess.v1.storage_backend")
        if pkg is not None:
            if not hasattr(pkg, "CreateStorageBackends"):
                pkg.CreateStorageBackends = None
            if not hasattr(pkg, "is_cuda_worker"):
                pkg.is_cuda_worker = lambda metadata: False
        from lmcache.inprocess.v1.storage_backend.storage_manager import (
            StorageManager,
        )

        manager = StorageManager.__new__(StorageManager)
        manager.local_cpu_backend = backend
        assert manager._host_free_bytes() == before
    finally:
        store.close()
        preload.close()
        LMCStatsMonitor.DestroyInstance()


def test_pool_sizes_match_surdata_direct_pools():
    """8GiB store and 12GiB preload. These are the fixed SurData sizes."""
    assert STORE_POOL_BYTES == 8589934592
    assert PRELOAD_POOL_BYTES == 12884901888
    assert STORE_POOL_BYTES + PRELOAD_POOL_BYTES == 20 * 1024**3


def test_preload_allocate_does_not_shrink_the_store_pool():
    """With two allocators, a preload allocate leaves the store free list."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="pre_evict_split",
    )
    store = MixedMemoryAllocator(1 << 20)
    preload = MixedMemoryAllocator(1 << 20)
    backend = LocalCPUBackend(config=config, memory_allocator=store)
    backend.preload_allocator = preload
    backend.store_allocator = store
    backend._split_pools = True
    try:
        before = backend._pool_free_stats(store, 4096)
        memory_obj = backend.allocate(
            _SHAPE, _DTYPE, eviction=False, purpose="preload"
        )
        assert memory_obj is not None
        assert getattr(memory_obj, "parent_allocator", None) is preload
        after = backend._pool_free_stats(store, 4096)
        assert after == before
        preload_after = backend._pool_free_stats(preload, 4096)
        assert preload_after is not None and before is not None
        assert preload_after[0] < before[0]
        memory_obj.ref_count_down()
    finally:
        store.close()
        preload.close()
        LMCStatsMonitor.DestroyInstance()


def test_store_watermark_does_not_evict_preload_objects():
    """Reclaim for D2H only removes can_evict objects that live in the store pool."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="pre_evict_split_evict",
    )
    store = MixedMemoryAllocator(256 * 1024)
    preload = MixedMemoryAllocator(1 << 20)
    backend = LocalCPUBackend(config=config, memory_allocator=store)
    backend.preload_allocator = preload
    backend.store_allocator = store
    backend._split_pools = True
    try:
        store_keys = _fill(backend)
        assert store_keys
        preload_obj = backend.allocate(
            _SHAPE, _DTYPE, eviction=False, purpose="preload"
        )
        assert preload_obj is not None
        preload_key = _key(10_000)
        backend.submit_put_task(preload_key, preload_obj)
        preload_obj.ref_count_down()
        backend.ensure_free_chunks(len(store_keys) + 1)
        assert preload_key in backend.hot_cache
        assert getattr(backend.hot_cache[preload_key], "parent_allocator", None) is preload
    finally:
        store.close()
        preload.close()
        LMCStatsMonitor.DestroyInstance()


def test_single_allocator_purpose_shares_the_pool():
    """purpose does nothing when the backend was given one allocator."""
    backend, allocator = _backend(1 << 20)
    try:
        before = backend._pool_free_stats(allocator, 4096)
        memory_obj = backend.allocate(
            _SHAPE, _DTYPE, eviction=False, purpose="preload"
        )
        assert memory_obj is not None
        after = backend._pool_free_stats(allocator, 4096)
        assert before is not None and after is not None
        assert after[0] < before[0]
        memory_obj.ref_count_down()
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_inflight_store_is_not_reclaimed_by_the_watermark():
    """A store object with ref_count > 1 stays. The watermark cannot take it."""
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 4
        held = resident[0]
        backend.hot_cache[held].ref_count_up()
        try:
            backend.ensure_free_chunks(len(resident))
            assert held in backend.hot_cache
            assert not backend.hot_cache[held].can_evict
        finally:
            backend.hot_cache[held].ref_count_down()
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_one_chunk_holes_add_up_and_admit_eight_allocs():
    """Separate one-chunk holes add. Eight later allocations fit without eviction.

    Every other chunk is freed and the rest are pinned, so the holes cannot
    be merged. ``store`` still allocates one chunk at a time, so the sum of
    those holes is the admission count.
    """
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 16
        for key in resident[::2]:
            assert backend.remove(key, force=True)
        for key in resident[1::2]:
            assert backend.hot_cache[key].pin()
        aligned = backend.aligned_store_slot_bytes()
        assert aligned > 0
        summed = backend._aligned_free_slots(aligned)
        slots = backend.ensure_free_chunks(8)
        assert summed is not None and summed >= 8
        assert slots == summed
        held = []
        for _ in range(8):
            memory_obj = backend.allocate(
                _SHAPE, _DTYPE, eviction=False, purpose="store"
            )
            assert memory_obj is not None
            held.append(memory_obj)
        for memory_obj in held:
            memory_obj.ref_count_down()
        for key in resident[1::2]:
            assert key in backend.hot_cache
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_seven_free_chunks_keep_evicting_until_eight():
    """Seven free slots are short of an eight-slot step, so eviction continues."""
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 8
        for key in resident[:7]:
            assert backend.remove(key, force=True)
        before = len(backend.hot_cache)
        slots = backend.ensure_free_chunks(8)
        assert slots >= 8
        assert len(backend.hot_cache) < before
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_watermark_reclaim_admits_store_when_the_slots_fit():
    """After reclaim, later one-chunk allocations fit in the returned slot count."""
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        resident = _fill(backend)
        assert len(resident) >= 4
        slot_bytes = backend.aligned_store_slot_bytes()
        span = backend.ensure_free_chunks(
            4, need_bytes=4 * slot_bytes, span_bytes=slot_bytes
        )
        assert span >= 4
        held = []
        for _ in range(4):
            memory_obj = backend.allocate(
                _SHAPE, _DTYPE, eviction=False, purpose="store"
            )
            assert memory_obj is not None
            held.append(memory_obj)
        for memory_obj in held:
            memory_obj.ref_count_down()
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()


def test_ensure_free_chunks_returns_zero_when_the_free_list_is_hidden():
    """No free list means the caller treats the pool as full and skips D2H."""
    backend, allocator = _backend(2 * 1024 * 1024)
    try:
        _fill(backend)
        before = len(backend.hot_cache)

        class _BareAllocator:
            pass

        backend.memory_allocator = _BareAllocator()
        assert backend.ensure_free_chunks(4) == 0
        assert len(backend.hot_cache) == before
    finally:
        allocator.close()
        LMCStatsMonitor.DestroyInstance()
