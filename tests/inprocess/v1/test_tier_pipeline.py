# SPDX-License-Identifier: Apache-2.0
"""Scheduler and worker queues for split metadata lookup."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import asyncio

# Third Party
import pytest

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.tier_pipeline import (
    PHASE_ABSENT,
    PHASE_MISS,
    PHASE_PENDING,
    PHASE_QUEUED,
    PHASE_READY,
    TierAdmission,
    TierLoad,
    TierWorkerQueue,
    admission_tokens,
    completion_kind,
    detach_tier_schedule_bridge,
    forget_preempt,
    install_tier_schedule_bridge,
    metadata_lookup_expired,
    offer_waiting_lookups,
)


def _load(num_chunks: int = 1) -> TierLoad:
    return TierLoad(
        num_chunks=num_chunks,
        tiers=[],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[num_chunks],
        keys_per_chunk=1,
    )


def test_config_defaults_off():
    config = LMCacheEngineConfig()
    assert config.split_metadata_lookup is False
    assert completion_kind(False) == ""
    assert completion_kind(True) == "preloaded"


def test_lookup_response_kind_defaults_empty():
    path = (
        Path(__file__).resolve().parents[3]
        / "lmcache/inprocess/v1/lookup_client/async_lookup_message.py"
    )
    spec = spec_from_file_location("async_lookup_message_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    msg = module.LookupResponseMsg(lookup_id="r", num_hit_tokens=4)
    assert msg.kind == ""


def test_admission_pending_is_not_a_miss_and_ready_is_positive():
    admission = TierAdmission()
    assert admission_tokens(admission.phase("r"), 0) is None

    assert admission.offer("r") is True
    assert admission.offer("r") is False
    assert admission.phase("r") == PHASE_PENDING
    assert admission_tokens(admission.phase("r"), 0) is None

    # Metadata hit only. The lookup window is free, HBM admission still waits.
    admission.note("r", "lookup", 128)
    assert admission.phase("r") == PHASE_QUEUED
    assert admission.inflight() == 0
    tokens = admission_tokens(admission.phase("r"), admission.preloaded_tokens("r"))
    assert tokens is None

    admission.note("r", "reading", 0)
    assert admission.phase("r") == PHASE_QUEUED

    admission.note("r", "preloaded", 128)
    assert admission.phase("r") == PHASE_READY
    tokens = admission_tokens(admission.phase("r"), admission.preloaded_tokens("r"))
    assert tokens == 128

    # A later ACK does not change a finished hit.
    admission.note("r", "lookup", 0)
    assert admission.phase("r") == PHASE_READY


def test_admission_lookup_zero_is_a_miss():
    admission = TierAdmission()
    admission.offer("r")
    admission.note("r", "lookup", 0)
    assert admission.phase("r") == PHASE_MISS
    assert admission_tokens(admission.phase("r"), 0) == 0
    assert admission.inflight() == 0


def test_legacy_empty_kind_means_the_body_is_in_host():
    admission = TierAdmission()
    admission.offer("r")
    admission.note("r", "", 32)
    assert admission.phase("r") == PHASE_READY
    assert admission.preloaded_tokens("r") == 32


def test_worker_miss_does_not_keep_a_get_and_unfit_stays_queued():
    queue = TierWorkerQueue()
    accepted, replaced = queue.note_lookup("miss", 0, None)
    assert accepted is True
    assert replaced is None
    assert "miss" not in queue._entries
    assert queue.take_next_fitting(lambda _job: True) is None
    assert queue.has_waiting() is False

    job = _load()
    queue.note_lookup("hit", 4, job)
    assert queue.take_next_fitting(lambda _job: False) is None
    assert queue.has_waiting() is True
    assert queue.drop_if_waiting("hit") is job
    assert queue.has_waiting() is False


def test_worker_claims_the_oldest_prefix_that_fits():
    queue = TierWorkerQueue()
    large = _load(4)
    small = _load(1)
    queue.note_lookup("large", 16, large)
    queue.note_lookup("small", 4, small)

    claimed = queue.take_next_fitting(lambda job: job.num_chunks <= 1)
    assert claimed is not None
    assert claimed[0] == "small"
    assert queue.has_waiting() is True
    assert queue.smallest_waiting_chunks() == 4
    assert queue.drop_if_waiting("large") is large
    assert queue.is_reading("small") is True
    assert queue.smallest_waiting_chunks() is None


class _Request:
    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.all_token_ids = tokens
        self.num_preemptions = 0
        self.priority = 0


class _Client:
    def __init__(self):
        self.tier_admission = TierAdmission()
        self.sent: list[str] = []
        self.retired: list[str] = []

    def submit_lookup(self, token_ids, lookup_id, request_configs=None):
        del token_ids, request_configs
        if not self.tier_admission.offer(lookup_id):
            return False
        self.sent.append(lookup_id)
        return True

    def retire_lookup(self, lookup_id):
        self.tier_admission.retire(lookup_id)
        self.retired.append(lookup_id)


def test_offer_starts_lookup_before_hbm_and_does_not_pop_the_queue():
    waiting = [_Request("a", [1, 2]), _Request("b", [3, 4]), _Request("c", [5])]
    scheduler = SimpleNamespace(
        waiting=waiting, skipped_waiting=[], scheduler_config=None
    )
    client = _Client()

    def prepare(request):
        return list(request.all_token_ids), None

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen={})
    assert client.sent == ["a", "b"]
    assert waiting[0].request_id == "a"
    assert len(waiting) == 3
    assert client.tier_admission.phase("c") == "absent"

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen={})
    assert client.sent == ["a", "b"]


def _storage_manager_cls():
    pytest.importorskip("lmcache.lmcache_native")
    # First Party
    from lmcache.inprocess.v1.storage_backend.storage_manager import (
        StorageManager,
    )

    return StorageManager


def test_split_flag_requires_async_loading():
    storage_manager = _storage_manager_cls()

    def enabled_for(cfg) -> bool:
        manager = storage_manager.__new__(storage_manager)
        manager.config = cfg
        return manager._split_metadata_lookup_enabled()

    assert enabled_for(
        SimpleNamespace(split_metadata_lookup=True, enable_async_loading=True)
    )
    assert not enabled_for(
        SimpleNamespace(split_metadata_lookup=True, enable_async_loading=False)
    )
    assert not enabled_for(
        SimpleNamespace(split_metadata_lookup=False, enable_async_loading=True)
    )


class _Backend:
    def __init__(self, hits: int):
        self.hits = hits
        self.gets = 0
        self.unpinned: list = []

    async def batched_async_contains(self, lookup_id, keys, pin):
        del lookup_id, pin
        return self.hits if self.hits <= len(keys) else len(keys)

    def unpin(self, key):
        self.unpinned.append(key)

    def batched_get_non_blocking(self, lookup_id, keys, transfer_spec=None):
        del lookup_id, keys, transfer_spec
        self.gets += 1
        raise AssertionError("GET must not start for this test")


class _Ack:
    def __init__(self):
        self.responses = []
        self.epochs: list[int] = []

    def send_response_to_scheduler(self, lookup_id, num_hit_tokens, kind="", epoch=0):
        self.responses.append((lookup_id, num_hit_tokens, kind))
        self.epochs.append(int(epoch))


def _manager(split: bool, backend: _Backend):
    storage_manager = _storage_manager_cls()
    manager = storage_manager.__new__(storage_manager)
    manager.config = SimpleNamespace(
        split_metadata_lookup=split,
        enable_async_loading=True,
    )
    manager.tier_queue = TierWorkerQueue()
    manager.async_lookup_server = _Ack()
    manager._tier_abort = {}
    manager._tier_cancelled = {}
    manager._tier_open = set()
    manager._tier_epoch_inflight = {}
    manager.loop = SimpleNamespace(is_running=lambda: False)
    manager.get_active_storage_backends = lambda search_range=None: [("disk", backend)]
    return manager


def test_flag_off_miss_is_one_ack_and_does_not_get():
    backend = _Backend(0)
    manager = _manager(False, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 0, "")]
    assert backend.gets == 0


def test_flag_on_miss_acks_lookup_and_does_not_get():
    backend = _Backend(0)
    manager = _manager(True, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 0, "lookup")]
    assert backend.gets == 0
    assert manager.tier_queue.has_waiting() is False


def test_flag_on_hit_stays_queued_until_it_fits():
    backend = _Backend(2)
    manager = _manager(True, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 8, "lookup")]
    assert backend.gets == 0
    assert manager.tier_queue.has_waiting() is True
    assert manager.tier_queue.take_next_fitting(lambda _job: False) is None
    claimed = manager.tier_queue.take_next_fitting(lambda _job: True)
    assert claimed is not None
    assert claimed[0] == "r"
    assert claimed[1].num_chunks == 2


def test_preempt_cancels_pending_and_offers_again():
    waiting = [_Request("a", [1, 2])]
    scheduler = SimpleNamespace(
        waiting=waiting, skipped_waiting=[], scheduler_config=None
    )
    client = _Client()
    seen: dict[str, int] = {}

    def prepare(request):
        return list(request.all_token_ids), None

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen=seen)
    assert client.sent == ["a"]
    first = client.tier_admission.generation("a")
    assert client.tier_admission.phase("a") == PHASE_PENDING

    waiting[0].num_preemptions = 1
    waiting[0].all_token_ids = [9, 9]
    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen=seen)
    assert client.retired == ["a"]
    assert client.sent == ["a", "a"]
    assert client.tier_admission.phase("a") == PHASE_PENDING
    assert client.tier_admission.generation("a") != first
    assert client.tier_admission.inflight() == 1

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen=seen)
    assert client.retired == ["a"]
    assert "a" in seen
    forget_preempt(seen, "a")
    assert "a" not in seen


def test_retired_generation_does_not_take_the_old_ack():
    admission = TierAdmission()
    assert admission.offer("r") is True
    first = admission.generation("r")
    admission.retire("r")
    assert admission.phase("r") == "absent"
    assert admission.offer("r") is True
    assert admission.generation("r") != first
    # The client drops an ACK whose epoch is not the current generation.
    # Applying it would mark the new lookup ready from the old prefix.
    if first == admission.generation("r"):
        admission.note("r", "preloaded", 128)
    assert admission.phase("r") == PHASE_PENDING
    assert admission_tokens(admission.phase("r"), 0) is None


def test_only_pending_metadata_expires():
    admission = TierAdmission()
    admission.offer("pending")
    offered = admission.offered_at("pending")
    assert offered is not None
    assert metadata_lookup_expired(PHASE_PENDING, offered, 10, offered + 0.02)
    assert not metadata_lookup_expired(PHASE_PENDING, offered, 10, offered)
    # What cancel_lookup does after the scheduler decides the lookup timed out.
    admission.note("pending", "lookup", 0)
    assert admission.phase("pending") == PHASE_MISS
    assert admission_tokens(admission.phase("pending"), 0) == 0

    admission.offer("queued")
    admission.note("queued", "lookup", 64)
    assert admission.phase("queued") == PHASE_QUEUED
    assert admission.inflight() == 0
    assert not metadata_lookup_expired(
        admission.phase("queued"), admission.offered_at("queued"), 1, offered + 100
    )
    assert admission_tokens(admission.phase("queued"), 0) is None


def test_preloaded_on_absent_is_not_ready():
    admission = TierAdmission()
    admission.note("r", "preloaded", 128)
    admission.note("r", "", 32)
    assert admission.phase("r") == "absent"
    assert admission_tokens(admission.phase("r"), 0) is None


def test_reading_before_lookup_leaves_the_window():
    admission = TierAdmission()
    admission.offer("r")
    assert admission.inflight() == 1
    admission.note("r", "reading", 0)
    assert admission.phase("r") == PHASE_QUEUED
    assert admission.inflight() == 0
    assert admission.queued() == 1
    # A reordered miss ACK must not undo a GET that has already started.
    admission.note("r", "lookup", 0)
    assert admission.phase("r") == PHASE_QUEUED
    assert admission_tokens(admission.phase("r"), 0) is None
    admission.mark_miss("r")
    assert admission.phase("r") == PHASE_MISS
    assert admission_tokens(admission.phase("r"), 0) == 0


def test_queued_fills_the_offer_window():
    waiting = [_Request("a", [1]), _Request("b", [2])]
    scheduler = SimpleNamespace(
        waiting=waiting, skipped_waiting=[], scheduler_config=None
    )
    client = _Client()
    seen: dict[str, int] = {}

    def prepare(request):
        return list(request.all_token_ids), None

    offer_waiting_lookups(scheduler, client, prepare, window=1, preempt_seen=seen)
    assert client.sent == ["a"]
    client.tier_admission.note("a", "lookup", 8)
    assert client.tier_admission.inflight() == 0
    assert client.tier_admission.queued() == 1
    offer_waiting_lookups(scheduler, client, prepare, window=1, preempt_seen=seen)
    assert client.sent == ["a"]


class _AbortBackend(_Backend):
    def __init__(self, manager_holder: dict):
        super().__init__(2)
        self.manager_holder = manager_holder

    async def batched_async_contains(self, lookup_id, keys, pin):
        manager = self.manager_holder["manager"]
        assert manager.abort_tier_load(lookup_id) is True
        return await super().batched_async_contains(lookup_id, keys, pin)


def test_abort_during_contains_does_not_get():
    holder: dict = {}
    backend = _AbortBackend(holder)
    manager = _manager(True, backend)
    holder["manager"] = manager
    manager.note_tier_lookup_started("r", 1)
    keys = [object(), object()]
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=keys, cum_chunk_lengths=[0, 4, 8], epoch=1
        )
    )
    assert backend.gets == 0
    assert backend.unpinned == keys
    assert manager.tier_queue.has_waiting() is False
    assert manager.async_lookup_server.responses == []
    assert manager._tier_abort == {}


def test_stale_cleanup_epoch_does_not_unpin():
    backend = _Backend(2)
    manager = _manager(True, backend)
    key = object()
    load = TierLoad(
        num_chunks=1,
        tiers=[(backend, [key], 1, [0, 4])],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[1],
        keys_per_chunk=1,
        epoch=2,
    )
    manager.tier_queue.note_lookup("r", 4, load)
    assert manager.abort_tier_load("r", 1) is True
    assert manager.tier_queue.has_waiting() is True
    assert backend.unpinned == []
    assert manager.abort_tier_load("r", 2) is True
    assert manager.tier_queue.has_waiting() is False
    assert backend.unpinned == [key]


def test_preloaded_zero_misses_queued_and_leaves_miss_and_absent():
    admission = TierAdmission()
    admission.offer("queued")
    admission.note("queued", "lookup", 8)
    assert admission.phase("queued") == PHASE_QUEUED
    admission.note("queued", "preloaded", 0)
    assert admission.phase("queued") == PHASE_MISS
    assert admission_tokens(admission.phase("queued"), None) == 0

    admission.note("queued", "preloaded", 0)
    assert admission.phase("queued") == PHASE_MISS

    assert admission.phase("absent") == PHASE_ABSENT
    admission.note("absent", "preloaded", 0)
    assert admission.phase("absent") == PHASE_ABSENT
    assert admission_tokens(admission.phase("absent"), None) is None


class _Events:
    def update_event_status(self, *args, **kwargs):
        return None


class _Mem:
    def __init__(self):
        self.is_pinned = True
        self.unpinned = False
        self.released = False

    def unpin(self):
        self.unpinned = True
        self.is_pinned = False

    def ref_count_down(self):
        self.released = True


def _failed_future(message: str):
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        future.set_exception(RuntimeError(message))
        return future
    finally:
        loop.close()


def test_aborted_read_acks_zero_preloaded_and_releases():
    backend = _Backend(0)
    manager = _manager(True, backend)
    manager.event_manager = _Events()
    key = object()
    mem = _Mem()
    load = TierLoad(
        num_chunks=1,
        tiers=[(backend, [key], 1, [0, 4])],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[1],
        keys_per_chunk=1,
        epoch=3,
    )
    manager.tier_queue.note_lookup("r", 4, load)
    assert manager.tier_queue.take_next_fitting(lambda _job: True) is not None
    manager._tier_abort["r"] = 3
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        future.set_result([[(key, mem)]])
        manager.prefetch_all_done_callback(
            future, "r", [0, 4], [1], keys_per_chunk=1, epoch=3
        )
    finally:
        loop.close()
    assert manager.tier_queue.is_reading("r") is False
    assert mem.unpinned is True
    assert mem.released is True
    assert manager.async_lookup_server.responses == [("r", 0, "preloaded")]
    assert manager.async_lookup_server.epochs == [3]


def test_failed_prefetch_acks_zero_only_when_split_is_on():
    reading = TierLoad(
        num_chunks=1,
        tiers=[],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[1],
        keys_per_chunk=1,
        epoch=2,
    )
    split = _manager(True, _Backend(0))
    split.event_manager = _Events()
    split.tier_queue.note_lookup("r", 4, reading)
    assert split.tier_queue.take_next_fitting(lambda _job: True) is not None
    split.prefetch_all_done_callback(
        _failed_future("split"), "r", [0, 4], [1], keys_per_chunk=1, epoch=2
    )
    assert split.tier_queue.is_reading("r") is False
    assert split.async_lookup_server.responses == [("r", 0, "preloaded")]
    assert split.async_lookup_server.epochs == [2]

    off = _manager(False, _Backend(0))
    off.event_manager = _Events()
    off.tier_queue.note_lookup("r", 4, reading)
    assert off.tier_queue.take_next_fitting(lambda _job: True) is not None
    with pytest.raises(RuntimeError, match="off"):
        off.prefetch_all_done_callback(
            _failed_future("off"), "r", [0, 4], [1], keys_per_chunk=1, epoch=2
        )
    assert off.tier_queue.is_reading("r") is False
    assert off.async_lookup_server.responses == []


def test_detach_restores_schedule_after_the_last_impl():
    seen: list[str] = []

    class Scheduler:
        @staticmethod
        def schedule(scheduler, *args, **kwargs):
            del scheduler, args, kwargs
            seen.append("original")
            return "ok"

    class Impl:
        def __init__(self, name: str):
            self.name = name

        def on_vllm_schedule(self, scheduler):
            del scheduler
            seen.append(self.name)

    first, second = Impl("a"), Impl("b")
    install_tier_schedule_bridge(Scheduler, first)
    install_tier_schedule_bridge(Scheduler, second)
    assert Scheduler.schedule._lmcache_tier_bridge is True
    assert Scheduler.schedule(SimpleNamespace()) == "ok"
    assert seen == ["a", "b", "original"]

    seen.clear()
    detach_tier_schedule_bridge(Scheduler, first)
    assert Scheduler.schedule._lmcache_tier_bridge is True
    Scheduler.schedule(SimpleNamespace())
    assert seen == ["b", "original"]

    original = Scheduler.schedule._lmcache_tier_original
    detach_tier_schedule_bridge(Scheduler, second)
    assert Scheduler.schedule is original
    seen.clear()
    assert Scheduler.schedule(SimpleNamespace()) == "ok"
    assert seen == ["original"]
    detach_tier_schedule_bridge(Scheduler, second)
    assert Scheduler.schedule is original


def test_lookup_while_reading_unpins_and_acks_miss():
    backend = _Backend(1)
    manager = _manager(True, backend)
    old_key = object()
    old = TierLoad(
        num_chunks=1,
        tiers=[(backend, [old_key], 1, [0, 4])],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[1],
        keys_per_chunk=1,
        epoch=1,
    )
    manager.tier_queue.note_lookup("r", 4, old)
    assert manager.tier_queue.take_next_fitting(lambda _job: True) is not None
    new_key = object()
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[new_key], cum_chunk_lengths=[0, 4], epoch=2
        )
    )
    assert backend.gets == 0
    assert backend.unpinned == [new_key]
    assert manager.tier_queue.is_reading("r") is True
    assert manager.async_lookup_server.responses == [("r", 0, "lookup")]
    assert manager.async_lookup_server.epochs == [2]


def test_cleanup_before_lookup_does_not_pin_or_queue():
    backend = _Backend(1)
    manager = _manager(True, backend)
    assert manager.abort_tier_load("r", 1) is False
    assert manager._tier_cancelled["r"] == {1}
    key = object()
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[key], cum_chunk_lengths=[0, 4], epoch=1
        )
    )
    assert backend.gets == 0
    assert backend.unpinned == []
    assert manager.async_lookup_server.responses == []
    assert manager.tier_queue.has_waiting() is False
    assert "r" not in manager._tier_cancelled

    later = _Backend(1)
    manager.get_active_storage_backends = lambda search_range=None: [("disk", later)]
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object()], cum_chunk_lengths=[0, 4], epoch=2
        )
    )
    assert later.gets == 0
    assert manager.async_lookup_server.responses == [("r", 4, "lookup")]
    assert manager.tier_queue.has_waiting() is True


def test_stale_cleanup_does_not_cancel_a_newer_contains():
    backend = _Backend(0)
    manager = _manager(True, backend)
    manager.note_tier_lookup_started("r", 2)
    assert manager.abort_tier_load("r", 1) is True
    assert manager._tier_abort == {}
    assert "r" in manager._tier_open


def test_empty_drain_does_not_rescan_when_rearmed():
    manager = _manager(True, _Backend(0))
    manager._tier_freed_since_scan = 0
    manager._tier_release_hold = 0
    manager._tier_drain_again = False
    manager.tier_queue.note_lookup("r", 32, _load(8))
    passes = {"n": 0}
    original = manager.tier_queue.take_next_fitting

    def counting(fits):
        passes["n"] += 1
        manager._tier_drain_again = True
        return original(fits)

    manager.tier_queue.take_next_fitting = counting
    manager._aligned_chunk_bytes = lambda: 100
    manager._host_free_bytes = lambda: 50
    asyncio.run(manager._tier_drain_once())
    assert passes["n"] == 1
    assert manager._tier_release_hold == 800
    assert manager._tier_drain_again is False
    assert manager._tier_freed_since_scan == 0
    assert manager.tier_queue.has_waiting() is True


def test_release_coalesces_until_the_smallest_prefix_fits():
    manager = _manager(True, _Backend(0))
    manager._tier_freed_since_scan = 0
    manager._tier_release_hold = 0
    manager._tier_drain_again = False
    manager.tier_queue.note_lookup("r", 32, _load(8))
    free = {"n": 50}
    manager._aligned_chunk_bytes = lambda: 100
    manager._host_free_bytes = lambda: free["n"]
    launched: list[str] = []
    manager._send_tier_ack = lambda *args, **kwargs: None
    manager._launch_tier_get = lambda lookup_id, load: launched.append(lookup_id)
    schedules = {"n": 0}

    def schedule():
        schedules["n"] += 1
        asyncio.run(manager._tier_drain_once())

    manager._schedule_tier_drain = schedule
    manager._note_host_released(100)
    assert schedules["n"] == 1
    assert manager._tier_release_hold == 800
    assert manager._tier_freed_since_scan == 0
    for _ in range(7):
        manager._note_host_released(100)
    assert schedules["n"] == 1
    assert manager._tier_freed_since_scan == 700
    free["n"] = 800
    manager._note_host_released(100)
    assert schedules["n"] == 2
    assert launched == ["r"]
    assert manager.tier_queue.is_reading("r") is True


def test_new_lookup_drains_while_a_release_hold_is_set():
    manager = _manager(True, _Backend(0))
    manager._tier_freed_since_scan = 0
    manager._tier_release_hold = 800
    manager._tier_drain_again = False
    schedules: list[str] = []
    manager._schedule_tier_drain = lambda: schedules.append("drain")
    manager.tier_queue.note_lookup("n", 4, _load(1))
    manager._schedule_tier_drain()
    assert schedules == ["drain"]
    manager._note_host_released(10)
    assert schedules == ["drain"]
    assert manager._tier_freed_since_scan == 10
