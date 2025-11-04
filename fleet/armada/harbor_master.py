"""
Harbor Master - Router/Triage Pattern

A Harbor Master routes incoming requests to the appropriate specialist captain,
like a harbor master directing ships to the correct dock.

Route → Specialist → Response
"""

from typing import Dict, Union, Optional, Callable
from fleet.captains.base_captain import BaseCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
from termcolor import colored
import logging

logger = logging.getLogger(__name__)


class HarborMaster:
    """
    Harbor Master implements the Router/Triage pattern.

    The Harbor Master analyzes incoming requests and routes them
    to the appropriate specialist captain.

    Perfect for:
    - Customer service / support systems
    - Multi-domain applications
    - Different expertise per request
    - Load balancing across agents
    """

    def __init__(
        self,
        specialists: Dict[str, BaseCaptain],
        name: str = "Harbor Master",
        description: str = "",
        routing_strategy: str = "llm",
        router_model: Optional[str] = None,
        router_provider: Optional[BaseProvider] = None,
        fallback_specialist: Optional[str] = None
    ):
        """
        Initialize a Harbor Master (Router).

        Args:
            specialists: Dict mapping category names to specialist captains
            name: Harbor Master name
            description: Description
            routing_strategy: "llm" (model-based) or "rules" (keyword-based)
            router_model: Model for LLM-based routing
            router_provider: Provider for routing decisions
            fallback_specialist: Default specialist if routing is unclear
        """
        self.specialists = specialists
        self.name = name
        self.description = description
        self.routing_strategy = routing_strategy
        self.router_model = router_model
        self.router_provider = router_provider
        self.fallback_specialist = fallback_specialist or list(specialists.keys())[0]

        # Statistics
        self.routing_stats = {spec: 0 for spec in specialists.keys()}
        self.routing_stats["total"] = 0

        logger.info(f"Initialized Harbor Master: {name} with {len(specialists)} specialists")

    def route(
        self,
        request: str,
        model: Optional[str] = None,
        show_routing: bool = True,
        **kwargs
    ) -> Payload:
        """
        Route a request to the appropriate specialist.

        Args:
            request: The incoming request
            model: Model to use (for specialist)
            show_routing: Whether to print routing decision
            **kwargs: Additional options for specialist

        Returns:
            Payload from the specialist with routing metadata
        """
        if show_routing:
            print(f"\n{'='*60}")
            print(colored(f"🏛️  {self.name}: Analyzing Request", 'white', attrs=['bold']))
            print(f"Request: {request[:80]}...")
            print(f"{'='*60}\n")

        # Determine which specialist to use
        specialist_name = self._determine_route(request)

        if specialist_name not in self.specialists:
            logger.warning(f"Unknown specialist '{specialist_name}', using fallback")
            specialist_name = self.fallback_specialist

        specialist = self.specialists[specialist_name]

        # Update stats
        self.routing_stats[specialist_name] += 1
        self.routing_stats["total"] += 1

        if show_routing:
            print(colored(f"🎯 Routing to: {specialist.name} ({specialist_name})", 'cyan'))
            print()

        # Route to specialist
        if hasattr(specialist, 'chat'):
            result = specialist.chat(request, model=model, **kwargs)
        else:
            result = specialist.send_message(request, model=model or "gpt-4o-mini", **kwargs)

        # Add routing metadata
        result.metadata["routed_by"] = self.name
        result.metadata["routed_to"] = specialist_name
        result.metadata["specialist_name"] = specialist.name
        result.metadata["routing_strategy"] = self.routing_strategy

        if show_routing:
            print(f"\n{'='*60}")
            print(colored(f"✓ Response from {specialist.name}", 'green'))
            print(f"{'='*60}\n")

        return result

    def _determine_route(self, request: str) -> str:
        """Determine which specialist should handle the request"""
        if self.routing_strategy == "llm":
            return self._llm_routing(request)
        elif self.routing_strategy == "rules":
            return self._rules_routing(request)
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def _llm_routing(self, request: str) -> str:
        """Use an LLM to determine routing"""
        if not self.router_provider:
            logger.warning("No router provider specified, falling back to rules-based routing")
            return self._rules_routing(request)

        # Create routing prompt
        specialist_descriptions = "\n".join([
            f"- {key}: {spec.description or spec.name}"
            for key, spec in self.specialists.items()
        ])

        routing_prompt = f"""You are a routing assistant. Analyze the following request and determine which specialist should handle it.

Available specialists:
{specialist_descriptions}

Request: {request}

Respond with ONLY the specialist key (e.g., "technical", "billing", etc.). No explanation needed."""

        try:
            # Use a simple chat captain for routing
            from fleet.captains.chat_captain import ChatCaptain
            router = ChatCaptain(
                provider=self.router_provider,
                system_prompt="You are a routing assistant. Respond only with the specialist key.",
                default_model=self.router_model or "gpt-4o-mini"
            )

            response = router.chat(routing_prompt)
            specialist_key = response.content.strip().lower()

            # Validate the key
            if specialist_key in self.specialists:
                return specialist_key
            else:
                logger.warning(f"LLM returned unknown specialist '{specialist_key}', using fallback")
                return self.fallback_specialist

        except Exception as e:
            logger.error(f"LLM routing failed: {e}, using fallback")
            return self.fallback_specialist

    def _rules_routing(self, request: str) -> str:
        """Use keyword-based rules for routing"""
        request_lower = request.lower()

        # Simple keyword matching
        # In a real implementation, this would be more sophisticated
        for specialist_key, specialist in self.specialists.items():
            # Check if specialist key appears in request
            if specialist_key.lower() in request_lower:
                return specialist_key

            # Check if specialist description keywords appear
            if specialist.description:
                desc_keywords = specialist.description.lower().split()
                if any(keyword in request_lower for keyword in desc_keywords):
                    return specialist_key

        # No match found, use fallback
        logger.info(f"No keyword match found, using fallback specialist")
        return self.fallback_specialist

    def add_specialist(self, key: str, specialist: BaseCaptain):
        """Add a new specialist to the harbor"""
        self.specialists[key] = specialist
        self.routing_stats[key] = 0
        logger.info(f"Added specialist '{key}': {specialist.name}")

    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics"""
        return self.routing_stats.copy()

    def __len__(self) -> int:
        return len(self.specialists)

    def __str__(self) -> str:
        return f"HarborMaster({self.name}, {len(self.specialists)} specialists)"

    def __repr__(self) -> str:
        return self.__str__()
