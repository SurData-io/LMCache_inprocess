# SPDX-License-Identifier: Apache-2.0
"""Split metadata lookup from host GET and from HBM admission.

Off by default. An empty lookup response still means the tokens are already
in host memory. When ``split_metadata_lookup`` is on, the scheduler and the
worker each keep an ordered queue:

* ``kind="lookup"`` records the hit length and does not read the body.
* ``kind="reading"`` means host GET has started and carries no length.
* ``kind="preloaded"`` means the body is in host memory.

A miss does not start a GET. A prefix that does not fit in host stays queued.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Optional
import threading
import time

STATUS_INIT = "INIT"
STATUS_READING = "READING"
STATUS_MISS = "MISS"

PHASE_ABSENT = "absent"
PHASE_PENDING = "pending"
PHASE_QUEUED = "queued"
PHASE_MISS = "miss"
PHASE_READY = "ready"


def admission_tokens(phase: str, preloaded: int) -> Optional[int]:
    """What ``get_num_new_matched_tokens`` may report. It does not start I/O.

    Pending, queued, and absent stay ``None`` so vLLM leaves the request in
    ``skipped_waiting``. Queued means the hit length is known and the body is
    not in host yet; that wait is not a miss. A confirmed miss is ``0``. A
    positive length is returned only after the body is in host memory.
    """
    if phase in (PHASE_ABSENT, PHASE_PENDING, PHASE_QUEUED):
        return None
    if phase == PHASE_MISS:
        return 0
    if phase == PHASE_READY:
        return max(0, int(preloaded))
    return None


def metadata_lookup_expired(
    phase: str,
    offered_at: Optional[float],
    timeout_ms: Any,
    now: float,
) -> bool:
    """True when a metadata lookup has had no ACK within ``timeout_ms``.

    A queued hit that is waiting for host memory does not expire. ``timeout_ms``
    is the existing ``lookup_timeout_ms`` setting.
    """
    if phase != PHASE_PENDING or offered_at is None:
        return False
    try:
        limit = float(timeout_ms)
    except (TypeError, ValueError):
        return False
    if limit <= 0:
        return False
    return (float(now) - float(offered_at)) * 1000 > limit


def completion_kind(split_enabled: bool) -> str:
    """Legacy completion has an empty kind. The split path says ``preloaded``."""
    return "preloaded" if split_enabled else ""


@dataclass
class TierLoad:
    """One prefix that already has a metadata hit and may still need a GET."""

    num_chunks: int
    tiers: list[tuple[Any, list[Any], int, list[int]]]
    cum_chunk_lengths_total: list[int]
    tier_expected_chunks: list[int]
    keys_per_chunk: int
    epoch: int = 0


class TierWorkerQueue:
    """Worker-side ordered queue. The oldest hit is read first if it fits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, Optional[TierLoad]]] = {}
        self._order: list[str] = []

    def note_lookup(
        self, req_id: str, hit_tokens: int, job: Optional[TierLoad]
    ) -> tuple[bool, Optional[TierLoad]]:
        """Record a metadata result.

        A non-positive hit becomes a miss and returns no GET job. Replacing a
        queued job returns the previous job so the caller can unpin it. A GET
        that has already started is left alone.
        """
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current[0] == STATUS_READING:
                return False, None
            replaced = current[1] if current is not None else None
            if current is not None:
                self._order = [item for item in self._order if item != key]
            if hit_tokens <= 0 or job is None:
                # A miss is not a queued GET. Drop the id so finished misses
                # do not accumulate for the life of the process.
                self._entries.pop(key, None)
                return True, replaced
            self._entries[key] = (STATUS_INIT, job)
            self._order.append(key)
            return True, replaced

    def take_next_fitting(
        self, fits: Callable[[TierLoad], bool]
    ) -> Optional[tuple[str, TierLoad]]:
        """Claim the oldest queued hit whose whole prefix fits. Leave the rest."""
        with self._lock:
            for key in self._order:
                status, job = self._entries.get(key, (STATUS_MISS, None))
                if status != STATUS_INIT or job is None:
                    continue
                if not fits(job):
                    continue
                self._entries[key] = (STATUS_READING, job)
                return key, job
            return None

    def has_waiting(self) -> bool:
        with self._lock:
            return any(
                status == STATUS_INIT and job is not None
                for status, job in self._entries.values()
            )

    def smallest_waiting_chunks(self) -> Optional[int]:
        """Smallest queued prefix, in chunks. None when nothing is waiting."""
        with self._lock:
            smallest: Optional[int] = None
            for status, job in self._entries.values():
                if status != STATUS_INIT or job is None or job.num_chunks <= 0:
                    continue
                if smallest is None or job.num_chunks < smallest:
                    smallest = int(job.num_chunks)
            return smallest

    def drop_if_waiting(self, req_id: str) -> Optional[TierLoad]:
        """Drop a hit that has not started GET. A started GET is left running."""
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is None or current[0] != STATUS_INIT:
                return None
            self._entries.pop(key, None)
            self._order = [item for item in self._order if item != key]
            return current[1]

    def waiting_epoch(self, req_id: str) -> Optional[int]:
        """Epoch of a hit that has not started GET. None when it is not waiting."""
        with self._lock:
            current = self._entries.get(str(req_id))
            if current is None or current[0] != STATUS_INIT or current[1] is None:
                return None
            return int(current[1].epoch)

    def is_reading(self, req_id: str) -> bool:
        with self._lock:
            current = self._entries.get(str(req_id))
            return current is not None and current[0] == STATUS_READING

    def reading_epoch(self, req_id: str) -> Optional[int]:
        with self._lock:
            current = self._entries.get(str(req_id))
            if current is None or current[0] != STATUS_READING or current[1] is None:
                return None
            return int(current[1].epoch)

    def finish(self, req_id: str) -> None:
        """Forget a GET that has completed or been discarded."""
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is None or current[0] != STATUS_READING:
                return
            self._entries.pop(key, None)
            self._order = [item for item in self._order if item != key]


class TierAdmission:
    """Scheduler-side phases, in the order lookups were offered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase: dict[str, str] = {}
        self._preloaded: dict[str, int] = {}
        self._order: list[str] = []
        self._generation: dict[str, int] = {}
        self._offered_at: dict[str, float] = {}

    def offer(self, req_id: str) -> bool:
        """Mark a new lookup pending. False if this id is already tracked."""
        key = str(req_id)
        with self._lock:
            phase = self._phase.get(key, PHASE_ABSENT)
            if phase != PHASE_ABSENT:
                return False
            self._generation[key] = self._generation.get(key, 0) + 1
            self._phase[key] = PHASE_PENDING
            self._offered_at[key] = time.monotonic()
            self._order.append(key)
            return True

    def generation(self, req_id: str) -> int:
        """Monotonic id of the lookup currently offered for this request."""
        with self._lock:
            return int(self._generation.get(str(req_id), 0))

    def offered_at(self, req_id: str) -> Optional[float]:
        with self._lock:
            return self._offered_at.get(str(req_id))

    def note(self, req_id: str, kind: str, tokens: int) -> None:
        """Apply one worker ACK. A later ACK does not revive a miss or a hit."""
        key = str(req_id)
        with self._lock:
            phase = self._phase.get(key, PHASE_ABSENT)
            if phase in (PHASE_MISS, PHASE_READY):
                return
            if kind == "reading":
                # GET has started, so the metadata wait is over even if the
                # lookup ACK has not been applied yet.
                if phase == PHASE_PENDING:
                    self._phase[key] = PHASE_QUEUED
                    self._offered_at.pop(key, None)
                return
            if kind == "lookup":
                if tokens <= 0:
                    if phase == PHASE_PENDING:
                        self._phase[key] = PHASE_MISS
                        self._offered_at.pop(key, None)
                elif phase == PHASE_PENDING:
                    self._phase[key] = PHASE_QUEUED
                    self._offered_at.pop(key, None)
                return
            if kind in ("preloaded", ""):
                # A body ACK for an id that is not in this lookup is ignored.
                # That includes a late ACK after clear_lookup_status.
                if phase not in (PHASE_PENDING, PHASE_QUEUED):
                    return
                self._offered_at.pop(key, None)
                if tokens <= 0:
                    self._phase[key] = PHASE_MISS
                else:
                    self._phase[key] = PHASE_READY
                    self._preloaded[key] = int(tokens)

    def mark_miss(self, req_id: str) -> None:
        """End a lookup the scheduler itself cancelled.

        A metadata ACK of zero does not do this once the phase is queued,
        because that ACK can arrive reordered behind a GET that already
        started. Cancellation still has to release the slot.
        """
        key = str(req_id)
        with self._lock:
            phase = self._phase.get(key, PHASE_ABSENT)
            if phase not in (PHASE_PENDING, PHASE_QUEUED):
                return
            self._phase[key] = PHASE_MISS
            self._offered_at.pop(key, None)

    def phase(self, req_id: str) -> str:
        with self._lock:
            return self._phase.get(str(req_id), PHASE_ABSENT)

    def preloaded_tokens(self, req_id: str) -> int:
        with self._lock:
            return int(self._preloaded.get(str(req_id), 0))

    def inflight(self) -> int:
        """Lookups that have been sent and have not received a metadata ACK."""
        with self._lock:
            return sum(1 for phase in self._phase.values() if phase == PHASE_PENDING)

    def queued(self) -> int:
        """Hits whose body is not in host memory yet. These hold prefix pins."""
        with self._lock:
            return sum(1 for phase in self._phase.values() if phase == PHASE_QUEUED)

    def clear(self, req_id: str) -> None:
        key = str(req_id)
        with self._lock:
            self._forget_locked(key)

    def retire(self, req_id: str) -> int:
        """Drop the phase and bump the generation so a late ACK cannot apply.

        The id is absent afterwards, so a new offer can start. The generation
        is not reset.
        """
        key = str(req_id)
        with self._lock:
            self._generation[key] = self._generation.get(key, 0) + 1
            self._forget_locked(key)
            return self._generation[key]

    def _forget_locked(self, key: str) -> None:
        self._phase.pop(key, None)
        self._preloaded.pop(key, None)
        self._offered_at.pop(key, None)
        self._order = [item for item in self._order if item != key]


def forget_preempt(preempt_seen: dict[str, int], req_id: str) -> None:
    """Drop one request's preemption count after that request has finished."""
    preempt_seen.pop(str(req_id), None)


def install_tier_schedule_bridge(scheduler_cls: Any, impl: Any) -> None:
    """Wrap ``scheduler_cls.schedule`` so each registered impl sees the pass.

    A second install only adds ``impl``. The original method stays on the
    wrapper until the last impl is detached.
    """
    impls = getattr(scheduler_cls, "_lmcache_tier_impls", None)
    if impls is None:
        impls = []
        scheduler_cls._lmcache_tier_impls = impls
    if impl not in impls:
        impls.append(impl)
    schedule = scheduler_cls.schedule
    if getattr(schedule, "_lmcache_tier_bridge", False):
        return

    def wrapped(scheduler, *args, **kwargs):
        for item in list(getattr(scheduler_cls, "_lmcache_tier_impls", ())):
            bridge = getattr(item, "on_vllm_schedule", None)
            if callable(bridge):
                bridge(scheduler)
        return schedule(scheduler, *args, **kwargs)

    wrapped._lmcache_tier_bridge = True  # type: ignore[attr-defined]
    wrapped._lmcache_tier_original = schedule  # type: ignore[attr-defined]
    scheduler_cls.schedule = wrapped


def detach_tier_schedule_bridge(scheduler_cls: Any, impl: Any) -> None:
    """Drop ``impl``. Restore the original schedule when it was the last one."""
    impls = getattr(scheduler_cls, "_lmcache_tier_impls", None)
    if not isinstance(impls, list) or impl not in impls:
        return
    impls.remove(impl)
    if impls:
        return
    schedule = getattr(scheduler_cls, "schedule", None)
    original = getattr(schedule, "_lmcache_tier_original", None)
    if callable(original):
        scheduler_cls.schedule = original


def offer_waiting_lookups(
    scheduler: Any,
    client: Any,
    prepare_tokens: Callable[[Any], Optional[tuple[Any, Optional[dict]]]],
    window: int,
    preempt_seen: dict[str, int],
) -> None:
    """Offer one metadata lookup per waiting request, capped by ``window``.

    Does not pop vLLM's waiting queue and does not read free HBM. ``window``
    is ``max_num_seqs``. A lookup that has not received its metadata ACK and a
    hit that is still waiting for host memory each hold one slot. The second
    kind is not a miss and does not use ``lookup_timeout_ms``.
    """
    admission = getattr(client, "tier_admission", None)
    submit = getattr(client, "submit_lookup", None)
    if admission is None or not callable(submit):
        return
    try:
        limit = max(1, int(window))
    except (TypeError, ValueError):
        limit = 1
    queued = getattr(admission, "queued", None)
    occupied = int(admission.inflight()) + (int(queued()) if callable(queued) else 0)
    for queue_name in ("waiting", "skipped_waiting"):
        waiting = getattr(scheduler, queue_name, None)
        if waiting is None:
            continue
        try:
            requests = list(waiting)
        except TypeError:
            heap = getattr(waiting, "_heap", None)
            requests = list(heap) if heap is not None else []
        for request in requests:
            if occupied >= limit:
                return
            req_id = str(getattr(request, "request_id", "") or "")
            if not req_id or req_id.startswith("mock_req"):
                continue
            preempt_n = int(getattr(request, "num_preemptions", 0) or 0)
            seen = int(preempt_seen.get(req_id, 0) or 0)
            phase = str(admission.phase(req_id))
            if preempt_n > seen and phase != PHASE_ABSENT:
                # The old prefix must not finish. retire bumps the generation
                # and leaves the id absent so this pass can offer again.
                retire = getattr(client, "retire_lookup", None)
                if callable(retire):
                    retire(req_id)
                else:
                    admission.retire(req_id)
                if phase in (PHASE_PENDING, PHASE_QUEUED):
                    occupied = max(0, occupied - 1)
                phase = PHASE_ABSENT
            if preempt_n > seen:
                preempt_seen[req_id] = preempt_n
            if phase in (PHASE_PENDING, PHASE_QUEUED, PHASE_READY, PHASE_MISS):
                continue
            prepared = prepare_tokens(request)
            if prepared is None:
                continue
            token_ids, request_configs = prepared
            # submit_lookup marks the id pending. A duplicate offer does not send.
            if submit(token_ids, lookup_id=req_id, request_configs=request_configs):
                occupied += 1
