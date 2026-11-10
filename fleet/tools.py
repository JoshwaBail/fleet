"""Tools: plain Python functions the model can call.

Any function works as a tool. Its name, docstring and type hints become the
schema the model sees; its arguments are validated before it runs.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_origin, get_type_hints, overload

from pydantic import ConfigDict, Field, create_model
from pydantic_core import to_json

from .context import Context

Handler = Callable[[Context[Any], dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Tool:
    """Something the model can call.

    Build one directly to expose a tool that is not a Python function (an MCP
    tool, a remote API): ``parameters`` is a JSON Schema object and ``handler``
    receives the run context and the arguments the model supplied.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


@overload
def tool(fn: Callable[..., Any], /, *, name: str | None = None, description: str | None = None) -> Tool: ...
@overload
def tool(*, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None, /, *, name: str | None = None, description: str | None = None
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a :class:`Tool`.

    Passing a bare function in ``Agent(tools=[...])`` does this for you; use
    ``tool`` explicitly (as a call or a decorator) to override the name or
    description.
    """
    if fn is None:
        return lambda f: tool(f, name=name, description=description)

    summary, param_docs = _parse_docstring(inspect.getdoc(fn) or "")
    hints = _type_hints(fn)
    fields: dict[str, Any] = {}
    context_param: str | None = None
    for param in inspect.signature(fn).parameters.values():
        annotation = hints.get(param.name, Any)
        if annotation is Context or get_origin(annotation) is Context:
            context_param = param.name
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{fn!r}: tools cannot take *args or **kwargs")
        default = ... if param.default is param.empty else param.default
        doc = param_docs.get(param.name)
        fields[param.name] = (annotation, Field(default, description=doc) if doc else Field(default))

    tool_name = name or getattr(fn, "__name__", None)
    if not tool_name:
        raise TypeError(f"{fn!r} has no __name__; pass name=")
    arguments_model = create_model(tool_name, __config__=ConfigDict(extra="forbid"), **fields)
    is_async = inspect.iscoroutinefunction(fn)

    async def handler(ctx: Context[Any], arguments: dict[str, Any]) -> Any:
        validated = arguments_model.model_validate(arguments)
        kwargs = {field_name: getattr(validated, field_name) for field_name in fields}
        if context_param:
            kwargs[context_param] = ctx
        if is_async:
            return await fn(**kwargs)
        # Blocking tools run in a worker thread so parallel calls stay parallel.
        return await asyncio.to_thread(fn, **kwargs)

    return Tool(tool_name, description or summary, strip_titles(arguments_model.model_json_schema()), handler)


def _type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(fn, include_extras=True)
    except NameError:
        # With postponed annotations, a type defined inside a function cannot be
        # looked up by name. That is fine for the context parameter, whose type
        # argument is never needed; anything else has to be resolvable.
        hints: dict[str, Any] = {}
        for name, annotation in inspect.get_annotations(fn).items():
            if isinstance(annotation, str) and re.match(r"(\w+\.)*Context\b", annotation):
                annotation = Context
            elif isinstance(annotation, str):
                try:
                    annotation = eval(annotation, getattr(fn, "__globals__", {}))
                except NameError as error:
                    raise TypeError(
                        f"Cannot resolve the type hint {annotation!r} of tool {getattr(fn, '__name__', fn)!r}; "
                        "types used in a tool's signature must be defined at module level."
                    ) from error
            hints[name] = annotation
        return hints


def render(value: Any) -> str:
    """How a tool's return value is shown to the model: strings as-is, anything else as JSON."""
    return value if isinstance(value, str) else to_json(value, fallback=str).decode()


_SECTION = re.compile(r"^(Args|Arguments|Parameters|Returns?|Raises|Yields?|Examples?|Notes?):\s*$")
_PARAM = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its summary and per-argument descriptions."""
    summary: list[str] = []
    params: dict[str, str] = {}
    section = current = indent = None
    for line in doc.splitlines():
        if match := _SECTION.match(line):
            section, current, indent = match[1], None, None
        elif section is None:
            summary.append(line)
        elif section in ("Args", "Arguments", "Parameters") and line.strip():
            depth = len(line) - len(line.lstrip())
            indent = depth if indent is None else indent
            entry = _PARAM.match(line.strip()) if depth <= indent else None
            if entry:
                current = entry[1]
                params[current] = entry[2]
            elif current:
                params[current] = f"{params[current]} {line.strip()}".strip()
    return "\n".join(summary).strip(), params


def strip_titles(schema: Any, *, is_schema: bool = True) -> Any:
    """Drop pydantic's auto-generated ``title`` keywords; they are noise in a prompt.

    ``is_schema`` is false for maps keyed by user-chosen names (``properties``,
    ``$defs``), where a key called "title" is a real field and must stay.
    """
    if isinstance(schema, list):
        return [strip_titles(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {
        key: strip_titles(value, is_schema=key not in ("properties", "$defs"))
        for key, value in schema.items()
        if not (is_schema and key == "title")
    }
