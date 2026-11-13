# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for LMCacheEngine.cleanup_memory_objs.

Some storage backends (LocalCPU, LocalDisk, P2P, Maru) pin the MemoryObj
they return via async prefetch, while others (Remote, Nixl, plugin tiers)
do not. cleanup_memory_objs must skip unpin for the latter to avoid
driving pin_count below zero.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import asyncio
import logging

# Third Party
import pytest

# First Party
from lmcache.inprocess.v1.cache_engine import LMCacheEngine
from lmcache.inprocess.v1.event_manager import (
    LOADING_EPOCH_ATTR,
    EventManager,
    EventStatus,
    EventType,
)
from tests.inprocess.utils import create_test_memory_obj


def test_cleanup_memory_objs_handles_mixed_pin_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Cleanup must unpin chunks that were pinned during prefetch and
    skip unpin for chunks that were not, leaving every chunk with a
    non-negative pin_count and no "Double unpin" warning.
    """
    pinned_obj = create_test_memory_obj()
    pinned_obj.pin()
    pinned_obj.ref_count_up()
    assert pinned_obj.metadata.pin_count == 1

    nonpinned_obj = create_test_memory_obj()
    assert nonpinned_obj.metadata.pin_count == 0

    future = MagicMock()
    future.result.return_value = [
        [(None, pinned_obj)],
        [(None, nonpinned_obj)],
    ]

    engine = SimpleNamespace(event_manager=MagicMock())
    engine.event_manager.pop_event_if_done.return_value = future

    caplog.set_level(logging.WARNING, logger="lmcache")

    LMCacheEngine.cleanup_memory_objs(engine, "test_lookup")  # type: ignore[arg-type]

    assert pinned_obj.metadata.pin_count == 0
    assert nonpinned_obj.metadata.pin_count == 0
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text

    pinned_obj.ref_count_down()


def _done_loading(lookup_id: str, memory_obj, epoch: int):
    """Put one pinned chunk into a real DONE loading event.

    Args:
        lookup_id: Event id cleanup and retrieve share.
        memory_obj: Chunk the event result holds.
        epoch: Lookup generation stamped on the future.

    Returns:
        The event manager that owns the future.
    """
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        future.set_result([[(None, memory_obj)]])
        setattr(future, LOADING_EPOCH_ATTR, epoch)
        manager = EventManager()
        manager.add_event(EventType.LOADING, lookup_id, future)
        manager.update_event_status(EventType.LOADING, lookup_id, EventStatus.DONE)
        return manager
    finally:
        loop.close()


def test_cleanup_ignores_a_different_epoch() -> None:
    """A stale cleanup must not free the prefetch a later retrieve still owns."""
    memory_obj = create_test_memory_obj()
    memory_obj.pin()
    pins = memory_obj.metadata.pin_count
    refs = memory_obj.get_ref_count()
    manager = _done_loading("req", memory_obj, epoch=2)
    engine = SimpleNamespace(event_manager=manager)

    LMCacheEngine.cleanup_memory_objs(engine, "req", epoch=1)  # type: ignore[arg-type]

    assert memory_obj.metadata.pin_count == pins
    assert memory_obj.get_ref_count() == refs
    assert manager.get_event_status(EventType.LOADING, "req") == EventStatus.DONE

    LMCacheEngine.cleanup_memory_objs(engine, "req", epoch=2)  # type: ignore[arg-type]
    LMCacheEngine.cleanup_memory_objs(engine, "req", epoch=2)  # type: ignore[arg-type]

    assert memory_obj.metadata.pin_count == 0
    assert manager.get_event_status(EventType.LOADING, "req") == EventStatus.NOT_FOUND
