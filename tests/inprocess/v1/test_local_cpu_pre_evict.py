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
    STORE_PREEVICT_CHUNK_CAP,
    LocalCPUBackend,
    aligned_free_slot_count,
    aligned_savable_tokens,
    chunks_for_save_batch,
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
