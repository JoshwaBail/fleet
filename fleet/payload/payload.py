"""
Payload - Response objects from Captain operations.

In Fleet, a Payload represents the cargo/data returned from a Captain's mission.
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Payload:
    """
    A Payload represents the response from a Captain's transmission.

    Attributes:
        content: The main response content
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        metadata: Additional metadata about the response
        timestamp: When the payload was created
        captain_name: Name of the captain who generated this payload
        tool_calls: Any tool calls made during generation
    """
    content: Any
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    captain_name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)"""
        return self.input_tokens + self.output_tokens

    @property
    def cost_estimate(self) -> float:
        """
        Estimate cost based on token usage.
        This is a rough estimate - actual costs vary by model.
        """
        # Rough estimates per 1M tokens (you'd want to customize this per model)
        input_cost_per_1m = 3.0  # $3 per 1M input tokens
        output_cost_per_1m = 15.0  # $15 per 1M output tokens

        input_cost = (self.input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (self.output_tokens / 1_000_000) * output_cost_per_1m

        return input_cost + output_cost

    def __str__(self) -> str:
        return f"Payload(content={str(self.content)[:100]}..., tokens={self.total_tokens})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class StreamingPayload:
    """
    A streaming payload for real-time responses.

    Useful for long-form content generation where you want to stream results.
    """
    content_chunks: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    is_complete: bool = False
    captain_name: Optional[str] = None

    def add_chunk(self, chunk: str):
        """Add a content chunk to the stream"""
        self.content_chunks.append(chunk)

    @property
    def content(self) -> str:
        """Get the full content as a single string"""
        return "".join(self.content_chunks)

    def to_payload(self) -> Payload:
        """Convert to a regular Payload once streaming is complete"""
        return Payload(
            content=self.content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            captain_name=self.captain_name,
            metadata={"streaming": True}
        )
