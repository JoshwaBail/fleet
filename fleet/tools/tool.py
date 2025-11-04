"""
Tool - Tool builder system for Fleet captains.

A Tool is a function that a Captain can use to accomplish tasks.
Tools provide capabilities like API calls, data access, and computations.
"""

from typing import Callable, Dict, Any, Optional, List, get_type_hints
from dataclasses import dataclass, field
from inspect import signature, Parameter
import json


@dataclass
class ToolParameter:
    """
    Represents a parameter for a Tool.

    Attributes:
        name: Parameter name
        type: Parameter type (string representation for JSON schema)
        description: What this parameter does
        required: Whether this parameter is required
        enum: Optional list of allowed values
        default: Default value if not required
    """
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Any = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


class Tool:
    """
    A Tool is a function that can be used by a Captain.

    This class provides a simple way to define tools with proper schemas
    that work across different LLM providers (OpenAI, Anthropic, etc.)
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[List[ToolParameter]] = None,
        auto_infer_params: bool = True
    ):
        """
        Create a new Tool.

        Args:
            name: Name of the tool (should be snake_case)
            description: What this tool does
            function: The actual function to execute
            parameters: List of parameters (if None, will try to infer)
            auto_infer_params: Whether to automatically infer parameters from function signature
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or []

        if auto_infer_params and not parameters:
            self.parameters = self._infer_parameters()

    def _infer_parameters(self) -> List[ToolParameter]:
        """
        Automatically infer parameters from function signature.
        """
        sig = signature(self.function)
        type_hints = get_type_hints(self.function)
        params = []

        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue

            # Try to infer type
            param_type = "string"  # default
            if param_name in type_hints:
                hint = type_hints[param_name]
                param_type = self._python_type_to_json_type(hint)

            # Check if required
            required = param.default == Parameter.empty

            # Get default value
            default = None if required else param.default

            params.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default=default
            ))

        return params

    def _python_type_to_json_type(self, python_type) -> str:
        """Convert Python type hints to JSON schema types"""
        type_mapping = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        # Handle basic types
        if python_type in type_mapping:
            return type_mapping[python_type]

        # Handle typing module types
        type_str = str(python_type)
        if "List" in type_str or "list" in type_str:
            return "array"
        elif "Dict" in type_str or "dict" in type_str:
            return "object"
        elif "int" in type_str:
            return "integer"
        elif "float" in type_str:
            return "number"
        elif "bool" in type_str:
            return "boolean"

        return "string"  # default fallback

    def execute(self, **kwargs) -> Any:
        """Execute the tool's function with given parameters"""
        return self.function(**kwargs)

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling schema.

        Returns a dict compatible with OpenAI's tools format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """
        Convert to Anthropic tool schema.

        Returns a dict compatible with Anthropic's tools format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def __str__(self) -> str:
        return f"Tool({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_function(cls, function: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'Tool':
        """
        Create a Tool from a function with automatic parameter inference.

        Args:
            function: The function to wrap as a tool
            name: Optional name (defaults to function name)
            description: Optional description (defaults to function docstring)

        Returns:
            A new Tool instance
        """
        tool_name = name or function.__name__
        tool_desc = description or function.__doc__ or f"Tool: {tool_name}"

        return cls(
            name=tool_name,
            description=tool_desc.strip(),
            function=function,
            auto_infer_params=True
        )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[ToolParameter]] = None
):
    """
    Decorator to easily create tools from functions.

    Usage:
        @tool(name="get_weather", description="Get weather for a location")
        def get_weather(location: str, units: str = "celsius") -> dict:
            # ... implementation
            return {"temp": 20, "condition": "sunny"}
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        return Tool(
            name=tool_name,
            description=tool_desc.strip(),
            function=func,
            parameters=parameters
        )

    return decorator


# Convenience builder function
def build_tool(
    name: str,
    description: str,
    parameters: List[ToolParameter],
    function: Callable
) -> Tool:
    """
    Builder function for creating tools.

    This is a more explicit way to create tools compared to the decorator.
    """
    return Tool(
        name=name,
        description=description,
        function=function,
        parameters=parameters,
        auto_infer_params=False
    )
