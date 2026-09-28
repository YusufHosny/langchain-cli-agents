# ABOUTME: LangChain chat model backed by `claude -p`, so code that expects .invoke /
# ABOUTME: .with_structured_output runs on a Claude subscription with no API key.
from __future__ import annotations

from typing import Any

from langchain_cli_agents.chat import ChatCLIBase
from langchain_cli_agents.claude.cli import ClaudeCLI
from langchain_cli_agents.core import CLIResult


class ChatClaudeCLI(ChatCLIBase):
  model: str = "sonnet"
  timeout: int = 600

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._cli = ClaudeCLI(model=self.model, timeout=self.timeout)

  @property
  def _llm_type(self) -> str:
    return "claude-cli"

  # "replace" is a pure chat model: our prompt only, no built-in tool definitions either
  def _call(self, prompt: str, system: str | None) -> CLIResult:
    return self._cli.invoke(
      prompt, system=system, system_mode=self.system_mode, allowed_tools=[], max_turns=1,
      builtin_tools=[] if self.system_mode == "replace" else None,
    )
