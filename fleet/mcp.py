"""Use the tools of an MCP server.

In keeping with the rest of fleet, you own the connection: connect with the
``mcp`` package however you like and pass the client in. fleet itself does not
import ``mcp``, and works with both its 1.x and 2.x releases.
"""

from __future__ import annotations

from typing import Any

from .context import Context
from .tools import Tool


async def mcp_tools(client: Any) -> list[Tool]:
    """List the tools of a connected MCP client (or ``ClientSession``) as fleet tools.

    Only the text of a tool's result reaches the model; other content types
    (images, embedded resources) are dropped.
    """
    return [_wrap(client, spec) for spec in (await client.list_tools()).tools]


def _wrap(client: Any, spec: Any) -> Tool:
    async def handler(ctx: Context[Any], arguments: dict[str, Any]) -> str:
        result = await client.call_tool(spec.name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        if _either(result, "is_error", "isError"):
            raise RuntimeError(text or "the MCP tool reported an error")
        return text

    return Tool(spec.name, spec.description or "", _either(spec, "input_schema", "inputSchema"), handler)


def _either(obj: Any, new: str, old: str) -> Any:
    """Read a field that mcp 2.x renamed from camelCase to snake_case."""
    return getattr(obj, new) if hasattr(obj, new) else getattr(obj, old)
