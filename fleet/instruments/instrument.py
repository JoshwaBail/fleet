"""
Instrument - Tool builder system for Fleet captains.

An Instrument is a tool that a Captain can use to accomplish tasks.
Think of it as equipment or gear that enhances a Captain's capabilities.
"""

from typing import Callable, Dict, Any, Optional, List, get_type_hints
from dataclasses import dataclass, field
from inspect import signature, Parameter
import json


@dataclass
class InstrumentParameter:
    """
    Represents a parameter for an Instrument.

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


class Instrument:
    """
    An Instrument is a tool that can be used by a Captain.

    This class provides a simple way to define tools with proper schemas
    that work across different LLM providers (OpenAI, Anthropic, etc.)
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[List[InstrumentParameter]] = None,
        auto_infer_params: bool = True
    ):
        """
        Create a new Instrument.

        Args:
            name: Name of the instrument (should be snake_case)
            description: What this instrument does
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

    def _infer_parameters(self) -> List[InstrumentParameter]:
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

            params.append(InstrumentParameter(
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
        """Execute the instrument's function with given parameters"""
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
        return f"Instrument({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


def instrument(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[InstrumentParameter]] = None
):
    """
    Decorator to easily create instruments from functions.

    Usage:
        @instrument(name="get_weather", description="Get weather for a location")
        def get_weather(location: str, units: str = "celsius") -> dict:
            # ... implementation
            return {"temp": 20, "condition": "sunny"}
    """
    def decorator(func: Callable) -> Instrument:
        instrument_name = name or func.__name__
        instrument_desc = description or func.__doc__ or f"Instrument: {instrument_name}"

        return Instrument(
            name=instrument_name,
            description=instrument_desc.strip(),
            function=func,
            parameters=parameters
        )

    return decorator


# Convenience builder function
def build_instrument(
    name: str,
    description: str,
    parameters: List[InstrumentParameter],
    function: Callable
) -> Instrument:
    """
    Builder function for creating instruments.

    This is a more explicit way to create instruments compared to the decorator.
    """
    return Instrument(
        name=name,
        description=description,
        function=function,
        parameters=parameters,
        auto_infer_params=False
    )
