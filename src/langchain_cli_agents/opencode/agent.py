# ABOUTME: OpenCodeAgent — a ToolAgent that answers by tool-calling an MCP endpoint through an
# ABOUTME: `opencode serve` configured with that remote MCP and a builtins-disabled agent.
from __future__ import annotations

import json
import threading

from langchain_cli_agents.agent import AgentAnswer, ToolAgent
from langchain_cli_agents.core import CLIResult, PauseGate, SystemMode
from langchain_cli_agents.mcp import MCPEndpoint
from langchain_cli_agents.opencode.server import (
  OpenCodeServer, agent_config, get_json, message_body, parse_message, send_message, split_model,
)

_AGENT_NAME = "langchain-cli-agents-tools"
_TOOL_PART_TYPES = ("tool", "tool-invocation")


# session history is a list of {info, parts} messages; flatten to one list of part dicts
def _flatten_parts(hist: object) -> list[dict]:
  if isinstance(hist, dict):
    hist = hist.get("messages") or hist.get("data") or []
  out: list[dict] = []
  for m in hist if isinstance(hist, list) else []:
    if not isinstance(m, dict):
      continue
    parts = m.get("parts")
    if isinstance(parts, list):
      out.extend(p for p in parts if isinstance(p, dict))
    elif m.get("type"):
      out.append(m)
  return out


# total chars of tool outputs the agent actually read; robust to a few opencode part shapes
def _read_chars(parts: list[dict]) -> int:
  total = 0
  for p in parts:
    if p.get("type") not in _TOOL_PART_TYPES:
      continue
    state = p.get("state") or {}
    out = state.get("output")
    if out is None:
      inv = p.get("toolInvocation") or {}
      out = (inv.get("result") or inv.get("output") or p.get("output")
             or p.get("result") or state.get("result"))
    if isinstance(out, str):
      total += len(out)
    elif isinstance(out, (dict, list)):
      total += len(json.dumps(out, default=str))
  return total


class OpenCodeAgent(ToolAgent):
  def __init__(
    self,
    model: str = "opencode/x-preview-f-free",
    timeout: int = 150,
    binary: str = "opencode",
    system_mode: SystemMode = "append",
    auto_pause: bool = True,
    backoff_seconds: float = 120.0,
    max_transient_retries: int = 8,
    variant: str | None = None,
  ) -> None:
    self.model = model
    self.provider_id, self.model_id = split_model(model)
    self.timeout = timeout
    self.binary = binary
    self.system_mode = system_mode
    self.auto_pause = auto_pause
    self.max_transient_retries = max_transient_retries
    self.variant = variant
    self.gate = PauseGate("opencode", backoff_seconds)
    self._lock = threading.Lock()
    self._server: OpenCodeServer | None = None
    self._endpoint: MCPEndpoint | None = None

  def _config(self, mcp: MCPEndpoint) -> dict:
    return {
      "mcp": {mcp.name: {"type": "remote", "url": mcp.url, "enabled": True}},
      "tools": {"*": False},
      "agent": {_AGENT_NAME: agent_config(self.system_mode,
                                          tools={mcp.opencode_tool_glob: True})},
    }

  # (re)start the serve whenever the endpoint changes (e.g. a memory is re-mounted)
  def _ensure_server(self, mcp: MCPEndpoint) -> OpenCodeServer:
    with self._lock:
      if self._server is None or self._endpoint != mcp:
        if self._server is not None:
          self._server.stop()
        self._server = OpenCodeServer(self._config(mcp), binary=self.binary)
        self._endpoint = mcp
      return self._server

  def stop(self) -> None:
    with self._lock:
      if self._server is not None:
        self._server.stop()
      self._server = None
      self._endpoint = None

  def answer(self, question: str, mcp: MCPEndpoint, system: str, max_turns: int) -> AgentAnswer:
    # opencode has no per-request turn cap; max_turns is accepted for ToolAgent compatibility
    server = self._ensure_server(mcp)
    body = message_body(self.provider_id, self.model_id, question, system=system,
                        agent=_AGENT_NAME, variant=self.variant)

    def _once() -> tuple[str, dict, CLIResult]:
      sid, msg = send_message(server.url(), body, self.timeout)
      return sid, msg, parse_message(sid, msg)

    sid, msg, res = self.gate.run(_once, auto_pause=self.auto_pause,
                                  max_transient_retries=self.max_transient_retries)

    parts = msg.get("parts", []) or []
    turns = sum(1 for p in parts if p.get("type") in (*_TOOL_PART_TYPES, "step-finish"))
    # tool outputs live in the full session history, not the final message
    read_chars = 0
    try:
      hist_parts = _flatten_parts(get_json(f"{server.url()}/session/{sid}/message"))
      read_chars = _read_chars(hist_parts)
      if turns == 0:
        turns = sum(1 for p in hist_parts if p.get("type") in _TOOL_PART_TYPES)
    except Exception:
      pass

    return AgentAnswer(
      text=res.text, input_tokens=res.total_input_tokens, output_tokens=res.output_tokens,
      cache_read=res.cache_read_tokens, cache_creation=res.cache_creation_tokens,
      cost_usd=res.cost_usd, num_turns=turns, read_chars=read_chars,
    )
