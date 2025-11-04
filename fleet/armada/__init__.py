"""Armada module - Multi-agent orchestration for Fleet"""

from fleet.armada.armada import Armada

# Advanced Patterns
from fleet.armada.harbor_master import HarborMaster
from fleet.armada.fleet_command import FleetCommand
from fleet.armada.council import Council

__all__ = [
    "Armada",
    # Advanced Patterns
    "HarborMaster",  # Router/Triage pattern
    "FleetCommand",  # Hierarchical pattern
    "Council",  # Debate pattern
]
