# ABOUTME: Mount methods of live Python objects behind an in-process HTTP MCP server (background
# ABOUTME: thread) and describe any MCP server as a typed MCPEndpoint the agent drivers consume.
from __future__ import annotations

import socket
import threading
import time
from contextlib import closing
from dataclasses import dataclass

DEFAULT_SERVER_NAME = "tools"


@dataclass(frozen=True)
class ToolSpec:
  name: str          # tool name the model sees
  method: str        # attribute on the bound object that implements it
  description: str


# what an agent needs to reach an MCP server; independent of who runs the server
@dataclass(frozen=True)
class MCPEndpoint:
  name: str
  url: str
  tool_names: tuple[str, ...]

  # claude code namespaces MCP tools as mcp__<server>__<tool>
  @property
  def claude_tools(self) -> list[str]:
    return [f"mcp__{self.name}__{t}" for t in self.tool_names]

  # opencode namespaces MCP tools as <server>_<tool>; this glob enables all of them
  @property
  def opencode_tool_glob(self) -> str:
    return f"{self.name}*"


@dataclass
class RunningMCP:
  endpoint: MCPEndpoint
  tools: list[ToolSpec]
  _server: object
  _thread: threading.Thread

  @property
  def url(self) -> str:
    return self.endpoint.url

  @property
  def allowed_tools(self) -> list[str]:
    return self.endpoint.claude_tools

  def stop(self) -> None:
    self._server.should_exit = True  # type: ignore[attr-defined]
    self._thread.join(timeout=10)


def _free_port() -> int:
  with closing(socket.socket()) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def serve_tools(
  bindings: list[tuple[object, list[ToolSpec]]],
  *, name: str = DEFAULT_SERVER_NAME, host: str = "127.0.0.1", startup_timeout: float = 60.0,
) -> RunningMCP:
  # deferred so the core harness works without the [mcp] extra installed
  import uvicorn
  from fastmcp import FastMCP

  mcp = FastMCP(name)
  tools: list[ToolSpec] = []
  for obj, specs in bindings:
    for spec in specs:
      mcp.tool(getattr(obj, spec.method), name=spec.name, description=spec.description)
      tools.append(spec)

  port = _free_port()
  server = uvicorn.Server(uvicorn.Config(mcp.http_app(path="/mcp"), host=host, port=port,
                                         log_level="warning"))
  thread = threading.Thread(target=server.run, daemon=True)
  thread.start()

  deadline = time.time() + startup_timeout
  while time.time() < deadline and not server.started:
    if not thread.is_alive():
      raise RuntimeError("MCP server thread died on startup")
    time.sleep(0.1)

  endpoint = MCPEndpoint(name=name, url=f"http://{host}:{port}/mcp/",
                         tool_names=tuple(t.name for t in tools))
  return RunningMCP(endpoint=endpoint, tools=tools, _server=server, _thread=thread)


def serve_object(obj: object, tools: list[ToolSpec], **kwargs) -> RunningMCP:
  return serve_tools([(obj, tools)], **kwargs)
