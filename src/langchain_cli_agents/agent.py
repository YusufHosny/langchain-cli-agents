# ABOUTME: Provider-agnostic contract for a multi-turn tool-calling agent: answer one question
# ABOUTME: using the tools of an MCP endpoint, returning a usage-tracked AgentAnswer.
from __future__ import annotations

import abc
from dataclasses import dataclass

from langchain_cli_agents.mcp import MCPEndpoint


@dataclass
class AgentAnswer:
  text: str
  input_tokens: int = 0     # total, including cache reads/writes
  output_tokens: int = 0
  cache_read: int = 0
  cache_creation: int = 0
  cost_usd: float = 0.0
  num_turns: int = 0
  read_chars: int = 0       # total chars of tool outputs the agent read (context-cost proxy)


class ToolAgent(abc.ABC):
  @abc.abstractmethod
  def answer(self, question: str, mcp: MCPEndpoint, system: str, max_turns: int) -> AgentAnswer:
    raise NotImplementedError("answer should be implemented by subclasses")
