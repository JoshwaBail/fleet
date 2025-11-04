"""Multi-agent orchestration for Fleet"""

from fleet.armada.agent_fleet import AgentFleet

# Advanced Patterns
from fleet.armada.router import Router
from fleet.armada.hierarchical_fleet import HierarchicalFleet
from fleet.armada.council import Council

__all__ = [
    "AgentFleet",  # Multi-agent orchestration (formerly Armada)
    # Advanced Patterns
    "Router",  # Router/Triage pattern (formerly HarborMaster)
    "HierarchicalFleet",  # Hierarchical pattern (formerly FleetCommand)
    "Council",  # Debate pattern
]
