"""
Council - Debate Pattern

A Council brings together agents with different perspectives to debate
and reach better conclusions through multi-perspective reasoning.

Agent 1 argues → Agent 2 argues → Agent 3 argues → Resolution
"""

from typing import List, Optional, Dict, Any
from fleet.captains.base_captain import BaseCaptain
from fleet.payload.payload import Payload
from termcolor import colored
import logging

logger = logging.getLogger(__name__)


class Council:
    """
    Council implements the Debate pattern.

    Multiple agents with different perspectives debate to reach
    better, more thoroughly considered conclusions.

    Perfect for:
    - Important decisions requiring scrutiny
    - Evaluating trade-offs
    - Challenging assumptions
    - Multi-perspective analysis
    """

    def __init__(
        self,
        debaters: List[BaseCaptain],
        name: str = "Council",
        description: str = "",
        moderator: Optional[BaseCaptain] = None,
        resolution_strategy: str = "synthesis",  # "synthesis", "vote", "consensus"
        max_rounds: int = 2
    ):
        """
        Initialize a Council (Debate orchestration).

        Args:
            debaters: List of captains with different perspectives
            name: Council name
            description: Description
            moderator: Optional moderator captain (uses first debater's provider if None)
            resolution_strategy: How to resolve the debate
            max_rounds: Maximum debate rounds
        """
        self.debaters = debaters
        self.name = name
        self.description = description
        self.moderator = moderator
        self.resolution_strategy = resolution_strategy
        self.max_rounds = max_rounds

        # Assign colors
        colors = ['red', 'blue', 'yellow', 'cyan', 'magenta', 'green']
        for i, debater in enumerate(debaters):
            if not hasattr(debater, 'color') or debater.color == 'white':
                debater.color = colors[i % len(colors)]

        logger.info(f"Initialized Council: {name} with {len(debaters)} debaters, {max_rounds} rounds")

    def convene(
        self,
        topic: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Payload:
        """
        Convene the council to debate a topic.

        Args:
            topic: The topic/question to debate
            model: Model to use for all debaters
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options

        Returns:
            Payload with resolution and debate transcript
        """
        print(f"\n{'='*60}")
        print(colored(f"🏛️  COUNCIL CONVENES: {self.name}", 'white', attrs=['bold']))
        print(f"Topic: {topic}")
        print(f"Debaters: {', '.join(d.name for d in self.debaters)}")
        print(f"{'='*60}\n")

        debate_transcript = []

        # Opening statements
        print(colored("📢 Round 1: Opening Statements", 'white', attrs=['bold']))
        print()

        opening_arguments = []
        for debater in self.debaters:
            print(colored(f"🗣️  {debater.name}:", debater.color, attrs=['bold']))

            debater.clear_messages()
            if hasattr(debater, 'chat'):
                argument = debater.chat(topic, model=model)
            else:
                argument = debater.send_message(topic, model, temperature, max_tokens, **kwargs)

            opening_arguments.append({
                'debater': debater.name,
                'round': 1,
                'argument': argument.content,
                'tokens': argument.total_tokens
            })

            print(f"{argument.content}\n")
            debate_transcript.append(f"[{debater.name}]: {argument.content}")

        # Rebuttal rounds
        for round_num in range(2, self.max_rounds + 1):
            print(colored(f"\n💬 Round {round_num}: Rebuttals", 'white', attrs=['bold']))
            print()

            round_arguments = []

            # Build context of previous arguments
            previous_args = "\n\n".join([
                f"{arg['debater']}: {arg['argument']}"
                for arg in opening_arguments[-len(self.debaters):]  # Last round
            ])

            for debater in self.debaters:
                print(colored(f"🗣️  {debater.name}:", debater.color, attrs=['bold']))

                rebuttal_prompt = f"""Topic: {topic}

Previous arguments from other council members:
{previous_args}

Respond to their arguments. Challenge weak points, build on strong points, and refine your position."""

                debater.clear_messages()
                if hasattr(debater, 'chat'):
                    rebuttal = debater.chat(rebuttal_prompt, model=model)
                else:
                    rebuttal = debater.send_message(rebuttal_prompt, model, temperature, max_tokens, **kwargs)

                round_arguments.append({
                    'debater': debater.name,
                    'round': round_num,
                    'argument': rebuttal.content,
                    'tokens': rebuttal.total_tokens
                })

                print(f"{rebuttal.content}\n")
                debate_transcript.append(f"[{debater.name}]: {rebuttal.content}")

            opening_arguments.extend(round_arguments)

        # Resolution phase
        print(colored(f"\n⚖️  Resolution Phase", 'white', attrs=['bold']))
        print()

        resolution = self._resolve_debate(
            topic,
            opening_arguments,
            model,
            temperature,
            max_tokens,
            **kwargs
        )

        print(colored(f"✓ Council Resolution:", 'green', attrs=['bold']))
        print(f"{resolution.content}\n")

        # Create final payload
        total_tokens = sum(arg['tokens'] for arg in opening_arguments) + resolution.total_tokens

        final_payload = Payload(
            content=resolution.content,
            input_tokens=resolution.input_tokens,
            output_tokens=resolution.output_tokens,
            metadata={
                "pattern": "Debate",
                "topic": topic,
                "debaters": [d.name for d in self.debaters],
                "rounds": self.max_rounds,
                "transcript": debate_transcript,
                "all_arguments": opening_arguments,
                "resolution_strategy": self.resolution_strategy,
                "total_tokens": total_tokens
            },
            captain_name=self.name
        )

        print(f"{'='*60}")
        print(colored(f"✓ COUNCIL ADJOURNED", 'green', attrs=['bold']))
        print(f"Total Tokens: {total_tokens}")
        print(f"{'='*60}\n")

        return final_payload

    def _resolve_debate(
        self,
        topic: str,
        arguments: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Resolve the debate based on the strategy"""

        if self.resolution_strategy == "synthesis":
            return self._synthesize_resolution(topic, arguments, model, temperature, max_tokens, **kwargs)
        elif self.resolution_strategy == "vote":
            return self._vote_resolution(topic, arguments, model, temperature, max_tokens, **kwargs)
        elif self.resolution_strategy == "consensus":
            return self._consensus_resolution(topic, arguments, model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown resolution strategy: {self.resolution_strategy}")

    def _synthesize_resolution(
        self,
        topic: str,
        arguments: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Synthesize all arguments into a balanced resolution"""

        # Format arguments
        formatted_args = []
        for arg in arguments:
            formatted_args.append(f"Round {arg['round']} - {arg['debater']}:\n{arg['argument']}")

        arguments_text = "\n\n".join(formatted_args)

        synthesis_prompt = f"""You are synthesizing a council debate.

Topic: {topic}

All arguments presented:
{arguments_text}

Synthesize these perspectives into a balanced, comprehensive conclusion that:
1. Acknowledges key points from each perspective
2. Identifies areas of agreement and disagreement
3. Provides a nuanced final recommendation
4. Notes any important caveats or considerations

Be fair to all perspectives while providing a clear conclusion."""

        # Use moderator if available, otherwise use first debater
        synthesizer = self.moderator if self.moderator else self.debaters[0]

        synthesizer.clear_messages()
        if hasattr(synthesizer, 'chat'):
            return synthesizer.chat(synthesis_prompt, model=model)
        else:
            return synthesizer.send_message(synthesis_prompt, model, temperature, max_tokens, **kwargs)

    def _vote_resolution(
        self,
        topic: str,
        arguments: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Resolve by having debaters vote on best argument"""

        # Simple implementation: return most recent arguments aggregated
        latest_args = [arg for arg in arguments if arg['round'] == self.max_rounds]

        vote_text = f"Topic: {topic}\n\nFinal positions:\n\n"
        for arg in latest_args:
            vote_text += f"{arg['debater']}: {arg['argument']}\n\n"

        vote_text += "\nAll positions have merit. Consider each perspective when making your decision."

        synthesizer = self.moderator if self.moderator else self.debaters[0]

        return Payload(
            content=vote_text,
            input_tokens=0,
            output_tokens=0
        )

    def _consensus_resolution(
        self,
        topic: str,
        arguments: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Find common ground and consensus"""

        formatted_args = "\n\n".join([
            f"{arg['debater']}: {arg['argument']}"
            for arg in arguments
            if arg['round'] == self.max_rounds
        ])

        consensus_prompt = f"""Topic: {topic}

Final positions:
{formatted_args}

Identify the common ground and consensus points among these perspectives.
What do they agree on? Where is there shared understanding?
Provide a consensus statement that all perspectives could support."""

        synthesizer = self.moderator if self.moderator else self.debaters[0]

        synthesizer.clear_messages()
        if hasattr(synthesizer, 'chat'):
            return synthesizer.chat(consensus_prompt, model=model)
        else:
            return synthesizer.send_message(consensus_prompt, model, temperature, max_tokens, **kwargs)

    def __len__(self) -> int:
        return len(self.debaters)

    def __str__(self) -> str:
        return f"Council({self.name}, {len(self.debaters)} debaters)"
