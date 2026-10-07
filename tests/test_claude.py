# ABOUTME: Unit tests for the claude chat model and MCP agent: system-prompt mode threading and
# ABOUTME: tool restriction on top of openai_cli_agents' ClaudeCLI, with the subprocess mocked out.
import json
import subprocess

import pytest

from langchain_cli_agents.claude import ChatClaudeCLI, ClaudeCodeAgent
from langchain_cli_agents.mcp import MCPEndpoint


def _completed(stdout: str) -> subprocess.CompletedProcess:
  return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _ok(result: str = "hi") -> str:
  return json.dumps({"result": result, "is_error": False, "num_turns": 1,
                     "total_cost_usd": 0.01, "session_id": "s1",
                     "usage": {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 100,
                               "cache_creation_input_tokens": 7}})


@pytest.fixture
def captured(monkeypatch):
  calls: list[dict] = []

  def fake_run(cmd, **kwargs):
    calls.append({"cmd": cmd, **kwargs})
    return _completed(_ok())

  monkeypatch.setattr(subprocess, "run", fake_run)
  return calls


def _flag(cmd: list[str], flag: str) -> str:
  return cmd[cmd.index(flag) + 1]


def test_chat_default_is_pure_model_without_system_prompt(captured):
  llm = ChatClaudeCLI()
  llm.invoke("hello")
  cmd = captured[0]["cmd"]
  assert _flag(cmd, "--system-prompt") == ""
  assert _flag(cmd, "--tools") == ""
  assert llm.usage_totals()["input_tokens"] == 117


def test_chat_replace_threads_system_message(captured):
  ChatClaudeCLI().invoke([("system", "sys"), ("human", "hello")])
  assert _flag(captured[0]["cmd"], "--system-prompt") == "sys"
  assert captured[0]["input"] == "hello"


def test_chat_append_keeps_builtin_tools(captured):
  ChatClaudeCLI(system_mode="append").invoke([("system", "sys"), ("human", "hello")])
  cmd = captured[0]["cmd"]
  assert _flag(cmd, "--append-system-prompt") == "sys"
  assert "--tools" not in cmd and "--system-prompt" not in cmd


def test_agent_restricts_to_endpoint_tools(captured):
  mcp = MCPEndpoint(name="mem", url="http://x/mcp/", tool_names=("grep",))
  ans = ClaudeCodeAgent().answer("q", mcp, "sys", max_turns=3)
  cmd = captured[0]["cmd"]
  assert _flag(cmd, "--allowed-tools") == "mcp__mem__grep"
  assert json.loads(_flag(cmd, "--mcp-config"))["mcpServers"]["mem"]["url"] == mcp.url
  assert _flag(cmd, "--system-prompt") == "sys" and _flag(cmd, "--tools") == ""
  assert ans.text == "hi" and ans.input_tokens == 117


def test_agent_append_keeps_claude_code_prompt(captured):
  mcp = MCPEndpoint(name="mem", url="http://x/mcp/", tool_names=("grep",))
  ClaudeCodeAgent(system_mode="append").answer("q", mcp, "sys", max_turns=3)
  cmd = captured[0]["cmd"]
  assert _flag(cmd, "--append-system-prompt") == "sys" and "--tools" not in cmd
