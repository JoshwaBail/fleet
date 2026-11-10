from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from conftest import Script, call, say

from fleet import Agent
from fleet.mcp import mcp_tools

pytest.importorskip("mcp")


def stock(item: str) -> str:
    """How many of an item are in stock."""
    if item != "apple":
        raise ValueError(f"unknown item: {item}")
    return "3 apples"


@asynccontextmanager
async def connected_client():
    """An MCP client talking to an in-process server exposing ``stock``, on mcp 1.x or 2.x."""
    try:
        from mcp.client import Client
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        from mcp.shared.memory import create_connected_server_and_client_session

        server = FastMCP("inventory")
        server.tool()(stock)
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            yield session
    else:
        server = MCPServer("inventory")
        server.tool()(stock)
        async with Client(server) as client:
            yield client


async def test_mcp_tools_are_listed_and_called_through_the_client():
    async with connected_client() as client:
        tools = await mcp_tools(client)
        assert [(t.name, t.description) for t in tools] == [("stock", "How many of an item are in stock.")]
        assert tools[0].parameters["required"] == ["item"]

        script = Script(call("stock", {"item": "apple"}), call("stock", {"item": "pear"}), say("3 apples, no pears"))
        result = await Agent(client=script, model="m", tools=tools).run("What's in stock?")

    found, missing = result.messages[2].parts[0], result.messages[4].parts[0]
    assert (found.content, found.is_error) == ("3 apples", False)
    assert missing.is_error and "stock" in missing.content
