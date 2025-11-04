"""
Toolbox - Collection of Tools for Fleet captains.

A Toolbox stores and manages multiple Tools that a Captain can use.
"""

from typing import Dict, List, Any, Optional, Callable
from fleet.tools.tool import Tool
import logging

logger = logging.getLogger(__name__)


class Toolbox:
    """
    A Toolbox manages a collection of Tools.

    This makes it easy to create reusable toolsets for different types of tasks.
    For example: navigation_tools, communication_tools, data_analysis_tools
    """

    def __init__(self, name: str = "", description: str = ""):
        """
        Create a new Toolbox.

        Args:
            name: Name of this toolbox (e.g., "Navigation Tools", "Data Processing")
            description: What this toolbox is for
        """
        self.name = name
        self.description = description
        self._tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Tool) -> 'Toolbox':
        """
        Add a tool to the toolbox.

        Args:
            tool: The tool to add

        Returns:
            Self for method chaining
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already exists in toolbox '{self.name}'. Overwriting.")

        self._tools[tool.name] = tool
        logger.info(f"Added tool '{tool.name}' to toolbox '{self.name}'")
        return self

    def remove_tool(self, name: str) -> 'Toolbox':
        """
        Remove a tool from the toolbox.

        Args:
            name: Name of the tool to remove

        Returns:
            Self for method chaining
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Removed tool '{name}' from toolbox '{self.name}'")
        else:
            logger.warning(f"Tool '{name}' not found in toolbox '{self.name}'")
        return self

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Get a list of all tool names in this toolbox"""
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """Get all tools in this toolbox"""
        return list(self._tools.values())

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tools as OpenAI function schemas.

        Returns:
            List of OpenAI-compatible tool schemas
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_anthropic_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tools as Anthropic tool schemas.

        Returns:
            List of Anthropic-compatible tool schemas
        """
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def get_functions_dict(self) -> Dict[str, Callable]:
        """
        Get a dictionary mapping tool names to their functions.

        This is useful for executing the actual functions when called by an LLM.
        """
        return {name: tool.function for name, tool in self._tools.items()}

    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Result of the tool execution
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in toolbox '{self.name}'")

        tool = self._tools[name]
        logger.info(f"Executing tool '{name}' from toolbox '{self.name}'")
        return tool.execute(**kwargs)

    def merge(self, other: 'Toolbox', conflict_strategy: str = "overwrite") -> 'Toolbox':
        """
        Merge another toolbox into this one.

        Args:
            other: Another toolbox to merge
            conflict_strategy: How to handle conflicts ("overwrite", "skip", or "error")

        Returns:
            Self for method chaining
        """
        for name, tool in other._tools.items():
            if name in self._tools:
                if conflict_strategy == "overwrite":
                    logger.info(f"Overwriting tool '{name}' during merge")
                    self._tools[name] = tool
                elif conflict_strategy == "skip":
                    logger.info(f"Skipping duplicate tool '{name}' during merge")
                    continue
                elif conflict_strategy == "error":
                    raise ValueError(f"Tool '{name}' already exists in toolbox '{self.name}'")
            else:
                self._tools[name] = tool

        logger.info(f"Merged toolbox '{other.name}' into '{self.name}'")
        return self

    def __len__(self) -> int:
        """Return the number of tools in this toolbox"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool exists in this toolbox"""
        return name in self._tools

    def __str__(self) -> str:
        return f"Toolbox({self.name}, {len(self._tools)} tools)"

    def __repr__(self) -> str:
        return self.__str__()


class ToolboxBuilder:
    """
    Fluent builder for creating Toolboxes.

    This provides a clean, chainable interface for building toolboxes.
    """

    def __init__(self, name: str = "", description: str = ""):
        self._toolbox = Toolbox(name, description)

    def with_tool(self, tool: Tool) -> 'ToolboxBuilder':
        """Add a tool to the toolbox"""
        self._toolbox.add_tool(tool)
        return self

    def with_tools(self, tools: List[Tool]) -> 'ToolboxBuilder':
        """Add multiple tools to the toolbox"""
        for tool in tools:
            self._toolbox.add_tool(tool)
        return self

    def with_function(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[List] = None
    ) -> 'ToolboxBuilder':
        """Add a function directly as a tool"""
        tool = Tool(name, description, function, parameters)
        self._toolbox.add_tool(tool)
        return self

    def build(self) -> Toolbox:
        """Build and return the toolbox"""
        return self._toolbox


# Convenience function for creating toolboxes
def create_toolbox(name: str = "", description: str = "") -> ToolboxBuilder:
    """
    Create a new toolbox using the builder pattern.

    Usage:
        toolbox = create_toolbox("Navigation", "Tools for navigation")\
            .with_tool(compass)\
            .with_tool(map_reader)\
            .build()
    """
    return ToolboxBuilder(name, description)
