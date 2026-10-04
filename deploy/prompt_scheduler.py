"""Prompt scheduling utilities for closed-loop ReactiveBFM evaluation.

The scheduler is deliberately independent of MuJoCo and the planner so it can
be unit-tested in a CPU-only environment.  Events are ordered by elapsed time,
while events observed at the same control tick are resolved by priority:
interactive user input (100) > scripted schedule (50) > random schedule (10).
"""

from __future__ import annotations

import csv
import json
import queue
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


USER_PRIORITY = 100
SCRIPT_PRIORITY = 50
RANDOM_PRIORITY = 10


@dataclass(frozen=True)
class PromptEvent:
    """A prompt update, addressed in seconds from control-loop start."""

    time_s: float
    prompt: str
    source: str = "script"
    priority: int = SCRIPT_PRIORITY
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.time_s < 0.0:
            raise ValueError("Prompt event time_s must be non-negative")
        if not self.prompt.strip():
            raise ValueError("Prompt event prompt must be non-empty")


def load_prompt_events(path: str | Path) -> list[PromptEvent]:
    """Load ``[{time_s, prompt, priority?}]`` JSON schedule."""

    schedule_path = Path(path).expanduser().resolve()
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("events", [])
    if not isinstance(payload, list):
        raise ValueError(f"Prompt schedule must be a JSON list: {schedule_path}")
    events: list[PromptEvent] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Schedule item {index} is not an object")
        try:
            time_s = float(item["time_s"])
            prompt = str(item["prompt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid schedule item {index}: {item!r}") from exc
        priority = int(item.get("priority", SCRIPT_PRIORITY))
        source = str(item.get("source", "script"))
        events.append(PromptEvent(time_s, prompt, source, priority, index))
    return sorted(events, key=lambda event: (event.time_s, -event.priority, event.sequence))


def load_prompt_pool(path: str | Path) -> list[str]:
    """Read non-empty captions from a Hymotion-style CSV."""

    csv_path = Path(path).expanduser().resolve()
    prompts: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "caption" not in (reader.fieldnames or []):
            raise ValueError(f"Prompt CSV must contain a caption column: {csv_path}")
        for row in reader:
            prompt = str(row.get("caption", "")).strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"Prompt CSV contains no non-empty captions: {csv_path}")
    return prompts


def random_prompt_events(
    prompts: Iterable[str],
    *,
    start_s: float,
    interval_s: float,
    count: int,
    seed: int | None = None,
    initial_prompt: str | None = None,
) -> list[PromptEvent]:
    """Generate deterministic random prompt events at fixed intervals."""

    if interval_s <= 0.0:
        raise ValueError("random interval_s must be positive")
    if count < 0:
        raise ValueError("random event count must be non-negative")
    pool = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not pool and count:
        raise ValueError("random prompt pool is empty")
    rng = random.Random(seed)
    events: list[PromptEvent] = []
    previous = str(initial_prompt).strip() if initial_prompt is not None else None
    for index in range(count):
        choices = [prompt for prompt in pool if prompt != previous] or pool
        prompt = rng.choice(choices)
        previous = prompt
        events.append(
            PromptEvent(
                time_s=float(start_s + index * interval_s),
                prompt=prompt,
                source="random",
                priority=RANDOM_PRIORITY,
                sequence=index,
            )
        )
    return events


class PromptScheduler:
    """Merge scripted/random events and optional interactive user prompts."""

    def __init__(
        self,
        initial_prompt: str,
        events: Iterable[PromptEvent] = (),
        *,
        interactive: bool = False,
        user_hold_s: float = 10.0,
        input_stream=None,
    ) -> None:
        if user_hold_s < 0.0:
            raise ValueError("user_hold_s must be non-negative")
        self.current_prompt = str(initial_prompt).strip()
        self._events = sorted(list(events), key=lambda event: (event.time_s, -event.priority, event.sequence))
        self._index = 0
        self._queue: queue.Queue[str] = queue.Queue()
        self._user_lock_until = -1.0
        self._user_hold_s = float(user_hold_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if interactive:
            stream = input_stream or sys.stdin
            self._thread = threading.Thread(target=self._read_input, args=(stream,), daemon=True)
            self._thread.start()

    def _read_input(self, stream) -> None:
        while not self._stop.is_set():
            line = stream.readline()
            if not line:
                return
            prompt = line.strip()
            if prompt:
                self._queue.put(prompt)

    def close(self) -> None:
        self._stop.set()

    def submit_user_prompt(self, prompt: str) -> None:
        """Queue a user command for the next control tick.

        This is the programmatic equivalent of typing into the interactive
        stdin stream and is useful for a UI, RPC endpoint, or a test harness.
        The command is still applied by :meth:`poll`, so all prompt changes
        are serialized with the control loop.
        """

        prompt = str(prompt).strip()
        if prompt:
            self._queue.put(prompt)

    def poll(self, elapsed_s: float) -> list[PromptEvent]:
        """Apply all updates due at ``elapsed_s`` and return accepted events."""

        elapsed_s = float(elapsed_s)
        accepted: list[PromptEvent] = []
        # User input is drained first, giving it the highest priority at this tick.
        while True:
            try:
                prompt = self._queue.get_nowait()
            except queue.Empty:
                break
            event = PromptEvent(elapsed_s, prompt, "user", USER_PRIORITY)
            self.current_prompt = event.prompt.strip()
            self._user_lock_until = elapsed_s + self._user_hold_s
            accepted.append(event)

        due: list[PromptEvent] = []
        while self._index < len(self._events) and self._events[self._index].time_s <= elapsed_s:
            due.append(self._events[self._index])
            self._index += 1
        # A command arriving on this tick wins even when user_hold_s=0.  Due
        # automatic events are consumed (rather than replayed after the hold)
        # so a user command cannot be immediately undone by a stale schedule.
        if due and (accepted or elapsed_s < self._user_lock_until):
            due = []
        if due:
            # Chronology wins across timestamps; priority resolves events at
            # the same timestamp (script beats random).
            event = max(due, key=lambda item: (item.time_s, item.priority, -item.sequence))
            self.current_prompt = event.prompt.strip()
            accepted.append(event)
        return accepted
