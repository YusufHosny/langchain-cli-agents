# langchain-cli-agents

A wrapper library to use coding agent CLI's in code.

| layer | Claude | opencode | what it is |
|---|---|---|---|
| raw CLI | `ClaudeCLI` | `OpenCodeCLI` | one stateless call → `CLIResult` (text, usage, cost) |
| LangChain | `ChatClaudeCLI` | `ChatOpenCodeCLI` | `BaseChatModel` with `.invoke` and prompt-enforced `.with_structured_output` |
| tool agent | `ClaudeCodeAgent` | `OpenCodeAgent` | multi-turn tool-calling over an MCP endpoint → `AgentAnswer` |

The raw CLI layer lives in [`openai-cli-agents`](https://github.com/YusufHosny/openai-cli-agents)
(re-exported here), which also serves it behind an OpenAI-compatible API.

`system_mode` defaults to `"replace"`: only your system prompt is sent, never the harness default
(claude code's prompt or built-in tools). Pass `system_mode="append"` for the pre-0.2 behaviour.

## Install

```bash
uv add "langchain-cli-agents[mcp] @ git+https://github.com/YusufHosny/langchain-cli-agents"
```

The `[mcp]` extra (fastmcp and uvicorn) is only needed for `serve_tools`/`serve_object`.

## Usage

```python
from langchain_cli_agents.claude import ChatClaudeCLI
from langchain_cli_agents.opencode import ChatOpenCodeCLI

llm = ChatClaudeCLI(model="sonnet")
llm.invoke([("system", "Answer in one word."), ("human", "Capital of France?")])
llm.with_structured_output(MyPydanticModel).invoke("...")
llm.usage_totals()   # {"input_tokens", "output_tokens", "cache_read", "cache_creation", "cost_usd"}

ChatOpenCodeCLI(model="opencode/big-pickle")
```

### Tool-calling agents over MCP

```python
from langchain_cli_agents import ToolSpec, serve_object
from langchain_cli_agents.claude import ClaudeCodeAgent

running = serve_object(my_obj, [ToolSpec("search", "search", "Search the notes.")], name="notes")
ans = ClaudeCodeAgent().answer("What did I note about X?", running.endpoint,
                               system="Use the tools.", max_turns=8)
running.stop()
```

`serve_tools` mounts methods of live Python objects behind an in-process HTTP MCP server. To use an MCP server you run yourself, pass `MCPEndpoint(name, url, tool_names)` instead.

