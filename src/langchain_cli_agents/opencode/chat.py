# ABOUTME: LangChain chat model backed by opencode (free / other-provider models), mirroring
# ABOUTME: ChatClaudeCLI so callers stay provider-agnostic.
from __future__ import annotations

from typing import Any

from langchain_cli_agents.chat import ChatCLIBase
from openai_cli_agents.core import CLIResult
from openai_cli_agents.opencode import OpenCodeCLI


class ChatOpenCodeCLI(ChatCLIBase):
  model: str = "opencode/muse-spark-1.2-contributor-free"
  timeout: int = 150
  variant: str | None = None

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._cli = OpenCodeCLI(model=self.model, timeout=self.timeout, variant=self.variant)

  @property
  def _llm_type(self) -> str:
    return "opencode-cli"

  def _call(self, prompt: str, system: str | None) -> CLIResult:
    return self._cli.invoke(prompt, system=system, system_mode=self.system_mode)
