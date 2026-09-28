# ABOUTME: Unit tests for the claude -p harness: flag building (system-prompt modes), output
# ABOUTME: parsing and limit detection, with subprocess mocked out.
import json
import subprocess

import pytest

from langchain_cli_agents.claude import ChatClaudeCLI, ClaudeCLI, ClaudeCodeAgent
from langchain_cli_agents.claude.cli import detect_limit
from langchain_cli_agents.core import HarnessError, LimitError
from langchain_cli_agents.mcp import MCPEndpoint


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
  return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _ok(result: str = "hi", **extra) -> str:
  return json.dumps({"result": result, "is_error": False, "num_turns": 1,
                     "total_cost_usd": 0.01, "session_id": "s1",
                     "usage": {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 100,
                               "cache_creation_input_tokens": 7}, **extra})


@pytest.fixture
def captured(monkeypatch):
  calls: list[dict] = []
  outputs: list[subprocess.CompletedProcess] = []

  def fake_run(cmd, **kwargs):
    calls.append({"cmd": cmd, **kwargs})
    return outputs.pop(0) if outputs else _completed(_ok())

  monkeypatch.setattr(subprocess, "run", fake_run)
  return calls, outputs


def test_append_is_default_system_mode():
  cmd = ClaudeCLI().build_cmd(system="be terse")
  assert cmd[cmd.index("--append-system-prompt") + 1] == "be terse"
  assert "--system-prompt" not in cmd


def test_replace_uses_system_prompt_flag():
  cmd = ClaudeCLI().build_cmd(system="be terse", system_mode="replace")
  assert cmd[cmd.index("--system-prompt") + 1] == "be terse"
  assert "--append-system-prompt" not in cmd


def test_builtin_tools_flag_only_when_set():
  assert "--tools" not in ClaudeCLI().build_cmd()
  cmd = ClaudeCLI().build_cmd(builtin_tools=[])
  assert cmd[cmd.index("--tools") + 1] == ""


def test_empty_allowed_tools_disables_all_tools():
  cmd = ClaudeCLI().build_cmd(allowed_tools=[])
  assert cmd[cmd.index("--allowed-tools") + 1] == ""


def test_extra_args_precede_variadic_flags():
  cmd = ClaudeCLI(extra_args=["--setting-sources", ""]).build_cmd(allowed_tools=["a", "b"])
  assert cmd.index("--setting-sources") < cmd.index("--allowed-tools")


def test_invoke_parses_usage_and_sends_prompt_on_stdin(captured):
  calls, _ = captured
  res = ClaudeCLI().invoke("question?")
  assert res.text == "hi"
  assert res.total_input_tokens == 117
  assert res.cost_usd == pytest.approx(0.01)
  assert calls[0]["input"] == "question?"


def test_session_limit_notice_in_result_raises_limit():
  assert detect_limit("You've hit your session limit · resets 3pm")[0]
  assert not detect_limit("the answer is 42")[0]


def test_limit_pauses_then_retries(captured, monkeypatch):
  _, outputs = captured
  outputs += [_completed(_ok("You've hit your usage limit")), _completed(_ok("fine"))]
  cli = ClaudeCLI(backoff_seconds=0.0)
  monkeypatch.setattr(cli.gate, "trigger", lambda reset_at: None)
  assert cli.invoke("q").text == "fine"


def test_limit_raises_without_auto_pause(captured):
  _, outputs = captured
  outputs.append(_completed(_ok("You've hit your usage limit")))
  with pytest.raises(LimitError):
    ClaudeCLI(auto_pause=False).invoke("q")


def test_nonzero_exit_without_stdout_is_harness_error(captured):
  _, outputs = captured
  outputs.append(_completed("", returncode=1, stderr="boom"))
  with pytest.raises(HarnessError, match="boom"):
    ClaudeCLI().invoke("q")


def test_chat_model_threads_system_mode(captured):
  calls, _ = captured
  llm = ChatClaudeCLI(system_mode="replace")
  llm.invoke([("system", "sys"), ("human", "hello")])
  cmd = calls[0]["cmd"]
  assert cmd[cmd.index("--system-prompt") + 1] == "sys"
  assert cmd[cmd.index("--tools") + 1] == ""
  assert llm.usage_totals()["input_tokens"] == 117


def test_chat_append_keeps_builtin_tools(captured):
  calls, _ = captured
  ChatClaudeCLI().invoke("hello")
  assert "--tools" not in calls[0]["cmd"]


def test_agent_restricts_to_endpoint_tools(captured):
  calls, _ = captured
  mcp = MCPEndpoint(name="mem", url="http://x/mcp/", tool_names=("grep",))
  ans = ClaudeCodeAgent(system_mode="replace").answer("q", mcp, "sys", max_turns=3)
  cmd = calls[0]["cmd"]
  assert cmd[cmd.index("--allowed-tools") + 1] == "mcp__mem__grep"
  assert json.loads(cmd[cmd.index("--mcp-config") + 1])["mcpServers"]["mem"]["url"] == mcp.url
  assert "--system-prompt" in cmd
  assert ans.text == "hi"
