# ABOUTME: Shared primitives for every harness: the result DTO, error hierarchy, system-prompt
# ABOUTME: mode, limit/transient text detection and the limit-triggered auto-pause gate.
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

# "append": keep the harness's own default system prompt and add ours after it.
# "replace": our prompt replaces the harness default (tool definitions still get injected).
SystemMode = Literal["append", "replace"]


@dataclass
class CLIResult:
  text: str
  is_error: bool
  usage: dict = field(default_factory=dict)
  cost_usd: float = 0.0
  num_turns: int = 0
  session_id: str | None = None
  raw: dict = field(default_factory=dict)

  @property
  def input_tokens(self) -> int:
    return int(self.usage.get("input_tokens", 0))

  @property
  def output_tokens(self) -> int:
    return int(self.usage.get("output_tokens", 0))

  @property
  def cache_read_tokens(self) -> int:
    return int(self.usage.get("cache_read_input_tokens", 0))

  @property
  def cache_creation_tokens(self) -> int:
    return int(self.usage.get("cache_creation_input_tokens", 0))

  # langchain convention: input_tokens is the total including cached reads/writes
  @property
  def total_input_tokens(self) -> int:
    return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


class HarnessError(RuntimeError):
  pass


class LimitError(HarnessError):
  def __init__(self, message: str, reset_at: float | None = None) -> None:
    super().__init__(message)
    self.reset_at = reset_at


# retryable provider blip that is not the model's fault (503, network_error, empty completion)
class TransientError(HarnessError):
  pass


def _lower_blob(texts: Iterable[str]) -> str:
  return " ".join(t for t in texts if t).lower()


def matches_any(patterns: Iterable[str], *texts: str) -> bool:
  blob = _lower_blob(texts)
  return any(p in blob for p in patterns)


def detect_limit(
  patterns: Iterable[str], parse_reset: Callable[[str], float | None], *texts: str,
) -> tuple[bool, float | None]:
  blob = _lower_blob(texts)
  if not any(p in blob for p in patterns):
    return False, None
  return True, parse_reset(blob)


def parse_epoch(blob: str) -> float | None:
  m = re.search(r"\b(1[6-9]\d{8}|20\d{8})\b", blob)
  return float(m.group(1)) if m else None


T = TypeVar("T")


# shared across one harness instance's concurrent callers: a limit hit by any caller
# pauses all of them until the reset time (or a fixed backoff when none is given)
class PauseGate:
  def __init__(self, label: str, backoff_seconds: float) -> None:
    self.label = label
    self.backoff_seconds = backoff_seconds
    self._lock = threading.Lock()
    self._resume_at = 0.0

  def wait(self) -> None:
    while True:
      with self._lock:
        remaining = self._resume_at - time.time()
      if remaining <= 0:
        return
      time.sleep(min(remaining, 30.0))

  def trigger(self, reset_at: float | None) -> None:
    now = time.time()
    until = reset_at if (reset_at and reset_at > now) else now + self.backoff_seconds
    with self._lock:
      if until <= self._resume_at:
        return
      self._resume_at = until
    print(f"[{self.label} limit] pausing ~{(until - now) / 60.0:.0f} min until "
          f"{time.strftime('%H:%M:%S', time.localtime(until))}", flush=True)

  # run fn, pausing and retrying on LimitError (up to max_limit_retries) and retrying
  # TransientError with a short linear backoff (up to max_transient_retries)
  def run(
    self, fn: Callable[[], T], *, auto_pause: bool = True,
    max_limit_retries: int = 1000, max_transient_retries: int = 0,
  ) -> T:
    limit_attempts = transient_attempts = 0
    while True:
      self.wait()
      try:
        return fn()
      except TransientError:
        transient_attempts += 1
        if transient_attempts > max_transient_retries:
          raise
        time.sleep(min(3 * transient_attempts, 20))
      except LimitError as e:
        limit_attempts += 1
        if not auto_pause or limit_attempts > max_limit_retries:
          raise
        self.trigger(e.reset_at)
