# ABOUTME: ClaudeCodeAgent — a ToolAgent that answers by letting `claude -p` tool-call an MCP
# ABOUTME: endpoint, restricted to that endpoint's tools.
from __future__ import annotations

from langchain_cli_agents.agent import AgentAnswer, ToolAgent
from langchain_cli_agents.claude.cli import ClaudeCLI
from langchain_cli_agents.core import SystemMode
from langchain_cli_agents.mcp import MCPEndpoint


class ClaudeCodeAgent(ToolAgent):
  def __init__(self, model: str = "sonnet", timeout: int = 300,
               system_mode: SystemMode = "append", cli: ClaudeCLI | None = None) -> None:
    self.cli = cli if cli is not None else ClaudeCLI(model=model, timeout=timeout)
    self.system_mode = system_mode

  def answer(self, question: str, mcp: MCPEndpoint, system: str, max_turns: int) -> AgentAnswer:
    res = self.cli.invoke(
      question, system=system, system_mode=self.system_mode,
      mcp_config={"mcpServers": {mcp.name: {"type": "http", "url": mcp.url}}},
      allowed_tools=mcp.claude_tools, max_turns=max_turns,
      # "replace": the endpoint's MCP tools are the only tools the model sees
      builtin_tools=[] if self.system_mode == "replace" else None,
    )
    return AgentAnswer(
      text=res.text, input_tokens=res.total_input_tokens, output_tokens=res.output_tokens,
      cache_read=res.cache_read_tokens, cache_creation=res.cache_creation_tokens,
      cost_usd=res.cost_usd, num_turns=res.num_turns,
    )
