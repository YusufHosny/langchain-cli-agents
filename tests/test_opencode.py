# ABOUTME: Unit tests for the opencode chat model and MCP agent: serve config per system-prompt
# ABOUTME: mode and tool-output accounting, on top of openai_cli_agents' OpenCodeCLI.
import pytest
from openai_cli_agents.opencode import cli as oc_cli
from openai_cli_agents.opencode.server import OpenCodeServerPool

from langchain_cli_agents.mcp import MCPEndpoint
from langchain_cli_agents.opencode import ChatOpenCodeCLI, OpenCodeAgent
from langchain_cli_agents.opencode.agent import _read_chars


def _msg(text: str = "ok") -> dict:
  return {"info": {"finish": "stop", "cost": 0.0,
                   "tokens": {"input": 3, "output": 2, "cache": {"read": 1, "write": 0}}},
          "parts": [{"type": "text", "text": text}]}


class _FakeServer:
  def url(self) -> str:
    return "http://fake"


class _FakePool(OpenCodeServerPool):
  def __init__(self) -> None:
    super().__init__()
    self.configs: list[dict] = []

  def get(self, config: dict):
    self.configs.append(config)
    return _FakeServer()


@pytest.fixture
def sent(monkeypatch):
  bodies: list[dict] = []

  def fake_send(base, body, timeout):
    bodies.append(body)
    return "sid", _msg()

  monkeypatch.setattr(oc_cli, "send_message", fake_send)
  return bodies


def _with_pool(llm: ChatOpenCodeCLI) -> _FakePool:
  pool = _FakePool()
  llm._cli.pool = pool
  return pool


def test_chat_default_uses_builtins_off_placeholder_agent(sent):
  llm = ChatOpenCodeCLI()
  pool = _with_pool(llm)
  assert llm.invoke("q").content == "ok"
  (agent_cfg,) = pool.configs[0]["agent"].values()
  assert agent_cfg["prompt"].strip() == "" and agent_cfg["tools"] == {"*": False}
  assert "system" not in sent[0]


def test_chat_append_uses_default_agent(sent):
  llm = ChatOpenCodeCLI(system_mode="append")
  pool = _with_pool(llm)
  llm.invoke([("system", "sys"), ("human", "q")])
  assert pool.configs == [{}]
  assert sent[0]["system"] == "sys" and "agent" not in sent[0]


def test_agent_config_restricts_tools_to_endpoint():
  mcp = MCPEndpoint(name="mem", url="http://x/mcp/", tool_names=("grep",))
  cfg = OpenCodeAgent()._config(mcp)
  assert cfg["mcp"]["mem"]["url"] == mcp.url
  assert cfg["tools"] == {"*": False}
  (agent_cfg,) = cfg["agent"].values()
  assert agent_cfg["tools"] == {"mem*": True} and "prompt" in agent_cfg
  (append_cfg,) = OpenCodeAgent(system_mode="append")._config(mcp)["agent"].values()
  assert "prompt" not in append_cfg


def test_read_chars_counts_tool_outputs():
  parts = [{"type": "tool", "state": {"output": "abcd"}},
           {"type": "tool-invocation", "toolInvocation": {"result": {"k": 1}}},
           {"type": "text", "text": "ignored"}]
  assert _read_chars(parts) == 4 + len('{"k": 1}')
