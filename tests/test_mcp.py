# ABOUTME: Integration test for serve_tools: mounts an object's method behind a real in-process
# ABOUTME: FastMCP HTTP server and calls it through the fastmcp client.
import asyncio

import pytest

from langchain_cli_agents.mcp import ToolSpec, serve_object

fastmcp = pytest.importorskip("fastmcp")


class _Echo:
  def shout(self, text: str) -> str:
    return text.upper()


def test_serve_object_exposes_named_tool():
  running = serve_object(_Echo(), [ToolSpec("shout", "shout", "uppercase text")], name="mem")
  try:
    assert running.allowed_tools == ["mcp__mem__shout"]
    assert running.endpoint.opencode_tool_glob == "mem*"

    async def _call() -> str:
      async with fastmcp.Client(running.url) as client:
        res = await client.call_tool("shout", {"text": "hi"})
        return res.content[0].text

    assert asyncio.run(_call()) == "HI"
  finally:
    running.stop()
