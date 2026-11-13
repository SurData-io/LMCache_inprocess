# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import Optional
import asyncio
import threading

# Attribute stamped on a LOADING future. Cleanup releases the buffers only
# when this is absent or equal to the cleanup epoch.
LOADING_EPOCH_ATTR = "lmcache_epoch"


class EventType(Enum):
    LOADING = auto()


class EventStatus(Enum):
    ONGOING = auto()
    DONE = auto()
    NOT_FOUND = auto()


class EventManager:
    """
    A thread-safe event manager to manage asynchronous events.
    Each event is identified by its type and a unique id.
    Events are organized by status for efficient counting.
    """

    def __init__(self) -> None:
        # Guard by lock
        # Structure: events[event_type][event_status][event_id] = future
        self.events: dict[EventType, dict[EventStatus, dict[str, asyncio.Future]]] = {
            et: {es: {} for es in EventStatus} for et in EventType
        }
        self.lock = threading.Lock()

    def add_event(
        self,
        event_type: EventType,
        event_id: str,
        future: asyncio.Future,
    ) -> None:
        """
        Add an event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            status_dict[EventStatus.ONGOING][event_id] = future

    def pop_event(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        """
        Pop and return the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            done_events = status_dict[EventStatus.DONE]
            assert event_id in done_events, (
                f"Event {event_id} of type {event_type} is not done or not found."
            )
            return done_events.pop(event_id)

    def update_event_status(
        self,
        event_type: EventType,
        event_id: str,
        status: EventStatus,
    ) -> None:
        """
        Update the status of the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            # Find the event in any status dict
            event = None
            for s in EventStatus:
                if event_id in status_dict[s]:
                    event = status_dict[s].pop(event_id)
                    break

            if event is None:
                raise KeyError(f"Event {event_id} of type {event_type} not found.")

            # Move to new status dict
            status_dict[status][event_id] = event

    def get_event_status(
        self,
        event_type: EventType,
        event_id: str,
    ) -> EventStatus:
        """
        Get the status of the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            for status in EventStatus:
                if event_id in status_dict[status]:
                    return status
            return EventStatus.NOT_FOUND

    def get_events_count_by_status(
        self,
        event_type: EventType,
        status: EventStatus,
    ) -> int:
        """
        Get the count of events for the given event type and status.
        This is a lightweight O(1) operation using dict length.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            if status_dict is None:
                return 0
            return len(status_dict[status])

    def pop_event_if_done(
        self,
        event_type: EventType,
        event_id: str,
        epoch: Optional[int] = None,
    ) -> Optional[asyncio.Future]:
        """Pop a DONE event, or leave it and return None.

        ``epoch`` None takes whatever is done. A finished prefetch and the
        cleanup that should release it then cannot both hold the future.
        When ``epoch`` is set and the future carries a different
        ``lmcache_epoch``, the event stays. A future with no epoch is
        eligible for any caller, which is the unstamped path.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            if status_dict is None:
                return None
            done_events = status_dict[EventStatus.DONE]
            future = done_events.get(event_id)
            if future is None:
                return None
            stamped = getattr(future, LOADING_EPOCH_ATTR, None)
            if (
                epoch is not None
                and stamped is not None
                and int(stamped) != int(epoch)
            ):
                return None
            return done_events.pop(event_id)

    def get_event_future(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        """
        Return the DONE event without removing it.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            done_events = status_dict[EventStatus.DONE]
            assert event_id in done_events, (
                f"Event {event_id} of type {event_type} is not done or not found."
            )
            return done_events[event_id]
