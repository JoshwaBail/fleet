"""
Quartermaster - Reflection Pattern

A Quartermaster uses the Reflection pattern: generating output, then
critiquing and improving it iteratively for higher quality.

Generate → Critique → Refine → Repeat
"""

from typing import Optional, List, Dict, Any
from fleet.captains.base_captain import BaseCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReflectionIteration:
    """Represents one iteration of the reflection process"""
    iteration: int
    content: str
    critique: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


class Quartermaster(BaseCaptain):
    """
    Quartermaster implements the Reflection pattern.

    The Quartermaster generates output, critiques it, and refines it
    through multiple iterations for higher quality results.

    Perfect for:
    - Writing and content creation
    - Code generation
    - When quality > speed
    - Self-improving outputs
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "Quartermaster",
        system_prompt: str = "You are a helpful AI assistant focused on producing high-quality outputs.",
        description: str = "",
        color: str = "yellow",
        critique_prompt: str = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048,
        max_iterations: int = 3,
        auto_improve: bool = True,
        verbose: bool = True
    ):
        """
        Initialize a Quartermaster (Reflection agent).

        Args:
            provider: The LLM provider
            name: Quartermaster name
            system_prompt: System instructions
            description: Quartermaster's role
            color: Terminal color
            critique_prompt: Custom critique instructions
            default_model: Default model
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
            max_iterations: Maximum reflection iterations
            auto_improve: Whether to auto-improve or wait for approval
            verbose: Whether to print iteration details
        """
        super().__init__(
            provider=provider,
            name=name,
            system_prompt=system_prompt,
            description=description,
            color=color
        )

        self.critique_prompt = critique_prompt or self._default_critique_prompt()
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_iterations = max_iterations
        self.auto_improve = auto_improve
        self.verbose = verbose

        logger.info(f"Initialized Quartermaster: {name} with Reflection pattern (max_iterations={max_iterations})")

    def _default_critique_prompt(self) -> str:
        """Default critique instructions"""
        return """Review the previous output critically and provide specific feedback on:
1. Clarity and coherence
2. Accuracy and completeness
3. Quality and effectiveness
4. Areas for improvement

Be constructive and specific in your critique."""

    def reflect(
        self,
        task: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_approval: bool = None,
        **kwargs
    ) -> Payload:
        """
        Generate output with reflection and iterative improvement.

        Args:
            task: The task/prompt to work on
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens
            require_approval: Override auto_improve setting
            **kwargs: Additional options

        Returns:
            Payload with final improved output and all iterations
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model specified and no default model set")

        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        require_approval = require_approval if require_approval is not None else not self.auto_improve

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📋 Quartermaster: {self.name}")
            print(f"Task: {task}")
            print(f"Max Iterations: {self.max_iterations}")
            print(f"{'='*60}\n")

        iterations: List[ReflectionIteration] = []

        # Initial generation
        if self.verbose:
            print(f"🎯 Iteration 1: Initial Generation")

        self.clear_messages()
        initial_response = self.send_message(
            content=task,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        iterations.append(ReflectionIteration(
            iteration=1,
            content=initial_response.content,
            critique=None,
            input_tokens=initial_response.input_tokens,
            output_tokens=initial_response.output_tokens
        ))

        if self.verbose:
            print(f"✓ Generated: {initial_response.content[:100]}...\n")

        current_content = initial_response.content

        # Reflection iterations
        for i in range(2, self.max_iterations + 1):
            if self.verbose:
                print(f"🔍 Iteration {i}: Critique & Refine")

            # Generate critique
            critique_response = self._critique(current_content, model, temperature, max_tokens, **kwargs)
            critique = critique_response.content

            if self.verbose:
                print(f"💬 Critique: {critique[:100]}...")

            # Check if we should continue
            if require_approval:
                if self.verbose:
                    print(f"\n⏸️  Waiting for approval to continue...")
                # In a real implementation, you'd have a callback here
                # For now, we'll continue automatically
                pass

            # Generate improved version
            improve_prompt = f"""Based on this critique:
{critique}

Please improve your previous response:
{current_content}

Provide an enhanced version addressing the feedback."""

            self.clear_messages()
            improved_response = self.send_message(
                content=improve_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            current_content = improved_response.content

            iterations.append(ReflectionIteration(
                iteration=i,
                content=current_content,
                critique=critique,
                input_tokens=improved_response.input_tokens,
                output_tokens=improved_response.output_tokens
            ))

            if self.verbose:
                print(f"✓ Improved: {current_content[:100]}...\n")

        # Create final payload
        total_input_tokens = sum(it.input_tokens for it in iterations)
        total_output_tokens = sum(it.output_tokens for it in iterations)

        final_payload = Payload(
            content=current_content,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            metadata={
                "pattern": "Reflection",
                "iterations": len(iterations),
                "all_iterations": [
                    {"iteration": it.iteration, "content": it.content, "critique": it.critique}
                    for it in iterations
                ]
            },
            captain_name=self.name
        )

        if self.verbose:
            print(f"{'='*60}")
            print(f"✓ Reflection Complete!")
            print(f"Total Iterations: {len(iterations)}")
            print(f"Total Tokens: {final_payload.total_tokens}")
            print(f"{'='*60}\n")

        return final_payload

    def _critique(
        self,
        content: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Generate a critique of the content"""
        critique_message = f"""{self.critique_prompt}

Content to review:
{content}"""

        self.clear_messages()
        return self.send_message(
            content=critique_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def __str__(self) -> str:
        return f"Quartermaster({self.name}, Reflection pattern, max_iterations={self.max_iterations})"
