from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from fleet import Context, Tool, tool
from fleet.tools import render


class Book(BaseModel):
    title: str
    pages: int = 0


@dataclass
class Point:
    x: int
    y: int


def test_schema_comes_from_the_signature_and_docstring():
    def search(query: str, limit: int = 10, mode: Literal["fast", "deep"] = "fast") -> list[str]:
        """Search the knowledge base.

        Use this before answering questions about internal docs.

        Args:
            query: What to look for.
            limit (int): Maximum number of results,
                between 1 and 50.
            mode: How hard to look.

        Returns:
            Matching titles.
        """
        return []

    made = tool(search)

    assert made.name == "search"
    assert made.description == "Search the knowledge base.\n\nUse this before answering questions about internal docs."
    assert made.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer", "default": 10, "description": "Maximum number of results, between 1 and 50."},
            "mode": {"type": "string", "enum": ["fast", "deep"], "default": "fast", "description": "How hard to look."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def test_name_and_description_can_be_overridden():
    @tool(name="lookup", description="Find a thing.")
    def find(thing: str) -> str:
        """Ignored."""
        return thing

    assert isinstance(find, Tool)
    assert (find.name, find.description) == ("lookup", "Find a thing.")


def test_annotated_descriptions_and_a_field_called_title():
    def shelve(book: Book, title: Annotated[str, Field(description="Shelf label.")]) -> str:
        return ""

    schema = tool(shelve).parameters
    assert schema["properties"]["title"] == {"type": "string", "description": "Shelf label."}
    assert schema["required"] == ["book", "title"]
    assert "title" in schema["$defs"]["Book"]["properties"]  # a real field named "title" survives
    assert "title" not in schema and "title" not in schema["$defs"]["Book"]  # the generated ones do not


async def test_arguments_are_validated_and_coerced():
    received = {}

    async def plot(point: Point, label: str = "p") -> str:
        received.update(point=point, label=label)
        return "ok"

    made = tool(plot)
    assert await made.handler(Context(), {"point": {"x": "1", "y": 2}}) == "ok"
    assert received == {"point": Point(1, 2), "label": "p"}

    with pytest.raises(ValidationError):
        await made.handler(Context(), {"point": {"x": 1}})
    with pytest.raises(ValidationError, match="Extra inputs"):
        await made.handler(Context(), {"point": {"x": 1, "y": 2}, "colour": "red"})


async def test_context_is_injected_wherever_it_is_declared():
    def lookup(key: str, ctx: Context[dict]) -> str:
        return ctx.deps[key]

    made = tool(lookup)
    assert list(made.parameters["properties"]) == ["key"]
    assert await made.handler(Context(deps={"a": "1"}), {"key": "a"}) == "1"


async def test_bound_methods_work_as_tools():
    class Store:
        def __init__(self) -> None:
            self.items = {"apple": 3}

        def count(self, item: str) -> int:
            """How many of an item are in stock."""
            return self.items[item]

    made = tool(Store().count)
    assert (made.name, list(made.parameters["properties"])) == ("count", ["item"])
    assert await made.handler(Context(), {"item": "apple"}) == 3


def test_unresolvable_type_hints_are_explained():
    class Local(BaseModel):
        x: int

    def use(thing: Local) -> None: ...

    with pytest.raises(TypeError, match="module level"):
        tool(use)


def test_var_args_are_rejected():
    def bad(*args: int) -> None: ...

    with pytest.raises(TypeError, match="args"):
        tool(bad)


def test_render():
    class Row(BaseModel):
        id: int

    assert render("plain") == "plain"
    assert render({"a": [1, 2]}) == '{"a":[1,2]}'
    assert render([Row(id=1)]) == '[{"id":1}]'
    assert render(None) == "null"
