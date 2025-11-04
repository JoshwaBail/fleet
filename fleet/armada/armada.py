"""
Armada - Multi-agent orchestration system.

An Armada coordinates multiple Captains to work together on complex missions.
Think of it as a fleet of ships working in concert.
"""

from typing import List, Union, Optional, Dict, Any
from fleet.captains.base_captain import BaseCaptain
from fleet.payload.payload import Payload
from termcolor import colored
import asyncio
import logging

logger = logging.getLogger(__name__)


class Armada:
    """
    Armada orchestrates multiple Captains working together.

    Supports two composition modes:
    - Sequential: Captains work one after another (chain of thought)
    - Parallel: Captains work simultaneously (diverse perspectives)
    """

    def __init__(
        self,
        captains: List[Union[BaseCaptain, 'Armada']],
        name: str = "Armada",
        description: str = "",
        synthesize: bool = True
    ):
        """
        Initialize an Armada.

        Args:
            captains: List of captains (or nested armadas) to orchestrate
            name: Name of this armada
            description: Mission description for this armada
            synthesize: Whether to synthesize results in parallel mode
        """
        self.captains = captains
        self.name = name
        self.description = description
        self.synthesize = synthesize

        # Assign colors for better visualization
        self.colors = ['magenta', 'cyan', 'yellow', 'green', 'blue', 'red']
        self._assign_colors()

        logger.info(f"Initialized Armada: {name} with {len(captains)} captains")

    def _assign_colors(self):
        """Assign colors to captains for terminal output"""
        for i, captain in enumerate(self.captains):
            if not hasattr(captain, 'color') or captain.color == 'white':
                captain.color = self.colors[i % len(self.colors)]

    def _log_action(self, captain: Union[BaseCaptain, 'Armada'], action: str, message: str, max_length: int = 100):
        """Log captain actions with color coding"""
        color = getattr(captain, 'color', 'white')
        truncated = message[:max_length] + "..." if len(message) > max_length else message
        print(colored(f"[{self.name}] {captain.name} - {action}: {truncated}", color))

    def voyage_sequential(
        self,
        initial_message: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Payload:
        """
        Execute a sequential voyage where captains work one after another.

        Each captain receives the accumulated context from previous captains.

        Args:
            initial_message: Starting message/prompt
            model: Model to use for all captains
            temperature: Sampling temperature
            max_tokens: Maximum tokens per captain
            **kwargs: Additional options

        Returns:
            Final payload from the last captain
        """
        print(colored(f"\n{'='*60}", 'white'))
        print(colored(f"[{self.name}] SEQUENTIAL VOYAGE STARTING", 'white', attrs=['bold']))
        print(colored(f"Mission: {initial_message[:100]}...", 'white'))
        print(colored(f"{'='*60}\n", 'white'))

        current_message = initial_message
        context_parts = [f"Original mission: {initial_message}"]
        final_payload = None

        for i, captain in enumerate(self.captains):
            self._log_action(captain, "⚡ ENGAGING", current_message)

            # Build context with all previous responses
            full_context = "\n\n".join(context_parts)

            # Get response
            if isinstance(captain, Armada):
                payload = captain.voyage_sequential(
                    full_context,
                    model,
                    temperature,
                    max_tokens,
                    **kwargs
                )
            else:
                payload = captain.send_message(
                    full_context,
                    model,
                    temperature,
                    max_tokens,
                    **kwargs
                )

            response_content = payload.content if payload.content else "[Tool calls only]"
            self._log_action(captain, "✓ COMPLETED", response_content)

            # Add this captain's response to context
            context_parts.append(f"{captain.name}'s analysis:\n{response_content}")

            final_payload = payload

        print(colored(f"\n[{self.name}] SEQUENTIAL VOYAGE COMPLETE\n", 'white', attrs=['bold']))
        return final_payload

    async def voyage_parallel(
        self,
        initial_message: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[Payload, List[Payload]]:
        """
        Execute a parallel voyage where captains work simultaneously.

        All captains receive the same initial message and work independently.
        Results can optionally be synthesized into a single response.

        Args:
            initial_message: Starting message/prompt
            model: Model to use for all captains
            temperature: Sampling temperature
            max_tokens: Maximum tokens per captain
            **kwargs: Additional options

        Returns:
            Synthesized payload or list of payloads
        """
        print(colored(f"\n{'='*60}", 'white'))
        print(colored(f"[{self.name}] PARALLEL VOYAGE STARTING", 'white', attrs=['bold']))
        print(colored(f"Mission: {initial_message[:100]}...", 'white'))
        print(colored(f"Deploying {len(self.captains)} captains simultaneously", 'white'))
        print(colored(f"{'='*60}\n", 'white'))

        async def captain_task(captain: Union[BaseCaptain, 'Armada']) -> Payload:
            """Execute a single captain's task"""
            self._log_action(captain, "⚡ LAUNCHING", initial_message)

            # Get response (in async context, we call sync methods)
            if isinstance(captain, Armada):
                # Nested armada uses sequential by default
                payload = captain.voyage_sequential(
                    initial_message,
                    model,
                    temperature,
                    max_tokens,
                    **kwargs
                )
            else:
                payload = captain.send_message(
                    initial_message,
                    model,
                    temperature,
                    max_tokens,
                    **kwargs
                )

            response_content = payload.content if payload.content else "[Tool calls only]"
            self._log_action(captain, "✓ RETURNED", response_content)

            return payload

        # Execute all captains in parallel
        tasks = [captain_task(captain) for captain in self.captains]
        payloads = await asyncio.gather(*tasks)

        print(colored(f"\n[{self.name}] All captains have returned", 'white', attrs=['bold']))

        # Synthesize if requested
        if self.synthesize and len(payloads) > 1:
            print(colored(f"[{self.name}] Synthesizing responses...", 'white'))
            synthesized = self._synthesize_payloads(payloads, model, temperature, max_tokens, **kwargs)
            print(colored(f"[{self.name}] PARALLEL VOYAGE COMPLETE (synthesized)\n", 'white', attrs=['bold']))
            return synthesized
        else:
            print(colored(f"[{self.name}] PARALLEL VOYAGE COMPLETE\n", 'white', attrs=['bold']))
            return payloads

    def _synthesize_payloads(
        self,
        payloads: List[Payload],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """
        Synthesize multiple payloads into a single coherent response.

        Args:
            payloads: List of payloads to synthesize
            model: Model to use for synthesis
            temperature: Temperature for synthesis
            max_tokens: Max tokens for synthesis
            **kwargs: Additional options

        Returns:
            Synthesized payload
        """
        # Format all captain responses
        responses_text = self._format_captain_responses(payloads)

        # Create synthesis prompt
        synthesis_prompt = f"""You are synthesizing the responses from multiple AI agents working on a mission.

Mission Description: {self.description if self.description else "Collaborative analysis"}

Each agent has provided their perspective below:

{responses_text}

Your task:
1. Identify key insights from each agent
2. Find common themes and patterns
3. Reconcile any contradictions
4. Synthesize into a single, coherent, comprehensive response
5. Ensure no important details are lost

Provide a well-structured synthesis that represents the collective intelligence of the team."""

        # Use the first captain's provider for synthesis
        first_captain = self.captains[0]
        while isinstance(first_captain, Armada):
            first_captain = first_captain.captains[0]

        provider = first_captain.provider

        # Create synthesis captain
        from fleet.captains.chat_captain import ChatCaptain
        synthesis_captain = ChatCaptain(
            provider=provider,
            name="Synthesis Captain",
            system_prompt="You are an expert at synthesizing multiple perspectives into coherent insights.",
            color='white'
        )

        self._log_action(synthesis_captain, "⚡ SYNTHESIZING", f"{len(payloads)} responses")

        # Get synthesis
        synthesis_payload = synthesis_captain.send_message(
            synthesis_prompt,
            model,
            temperature,
            max_tokens,
            **kwargs
        )

        self._log_action(synthesis_captain, "✓ SYNTHESIS COMPLETE", synthesis_payload.content)

        # Add metadata
        synthesis_payload.metadata["synthesized_from"] = len(payloads)
        synthesis_payload.metadata["armada"] = self.name

        return synthesis_payload

    def _format_captain_responses(self, payloads: List[Payload]) -> str:
        """Format captain responses for synthesis"""
        formatted = []

        for i, (captain, payload) in enumerate(zip(self.captains, payloads), 1):
            captain_name = captain.name
            captain_desc = getattr(captain, 'description', '')
            content = payload.content if payload.content else "[No text content]"

            formatted.append(f"""
{'='*50}
CAPTAIN {i}: {captain_name}
Role: {captain_desc if captain_desc else 'Not specified'}
{'='*50}

{content}
""")

        return "\n".join(formatted)

    def voyage(
        self,
        message: str,
        model: str,
        mode: str = "sequential",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[Payload, List[Payload]]:
        """
        Execute a voyage (mission) with this armada.

        Args:
            message: The mission message/prompt
            model: Model to use
            mode: "sequential" or "parallel"
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional options

        Returns:
            Payload(s) with results
        """
        if mode == "sequential":
            return self.voyage_sequential(message, model, temperature, max_tokens, **kwargs)
        elif mode == "parallel":
            return asyncio.run(
                self.voyage_parallel(message, model, temperature, max_tokens, **kwargs)
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'sequential' or 'parallel'")

    def add_captain(self, captain: Union[BaseCaptain, 'Armada']):
        """Add a captain to this armada"""
        self.captains.append(captain)
        self._assign_colors()
        logger.info(f"{self.name} - Added captain: {captain.name}")

    def __len__(self) -> int:
        return len(self.captains)

    def __str__(self) -> str:
        return f"Armada({self.name}, {len(self.captains)} captains)"

    def __repr__(self) -> str:
        return self.__str__()
