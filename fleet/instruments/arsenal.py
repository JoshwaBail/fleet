"""
Arsenal - Collection of Instruments for Fleet captains.

An Arsenal is like a cargo hold or equipment bay - it stores and manages
multiple Instruments that a Captain can use on their missions.
"""

from typing import Dict, List, Any, Optional, Callable
from fleet.instruments.instrument import Instrument
import logging

logger = logging.getLogger(__name__)


class Arsenal:
    """
    An Arsenal manages a collection of Instruments.

    This makes it easy to create reusable toolsets for different types of missions.
    For example: navigation_arsenal, communication_arsenal, data_analysis_arsenal
    """

    def __init__(self, name: str, description: str = ""):
        """
        Create a new Arsenal.

        Args:
            name: Name of this arsenal (e.g., "Navigation Tools", "Data Processing")
            description: What this arsenal is for
        """
        self.name = name
        self.description = description
        self._instruments: Dict[str, Instrument] = {}

    def add_instrument(self, instrument: Instrument) -> 'Arsenal':
        """
        Add an instrument to the arsenal.

        Args:
            instrument: The instrument to add

        Returns:
            Self for method chaining
        """
        if instrument.name in self._instruments:
            logger.warning(f"Instrument '{instrument.name}' already exists in arsenal '{self.name}'. Overwriting.")

        self._instruments[instrument.name] = instrument
        logger.info(f"Added instrument '{instrument.name}' to arsenal '{self.name}'")
        return self

    def remove_instrument(self, name: str) -> 'Arsenal':
        """
        Remove an instrument from the arsenal.

        Args:
            name: Name of the instrument to remove

        Returns:
            Self for method chaining
        """
        if name in self._instruments:
            del self._instruments[name]
            logger.info(f"Removed instrument '{name}' from arsenal '{self.name}'")
        else:
            logger.warning(f"Instrument '{name}' not found in arsenal '{self.name}'")
        return self

    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Get an instrument by name"""
        return self._instruments.get(name)

    def list_instruments(self) -> List[str]:
        """Get a list of all instrument names in this arsenal"""
        return list(self._instruments.keys())

    def get_all_instruments(self) -> List[Instrument]:
        """Get all instruments in this arsenal"""
        return list(self._instruments.values())

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all instruments as OpenAI function schemas.

        Returns:
            List of OpenAI-compatible tool schemas
        """
        return [instrument.to_openai_schema() for instrument in self._instruments.values()]

    def to_anthropic_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all instruments as Anthropic tool schemas.

        Returns:
            List of Anthropic-compatible tool schemas
        """
        return [instrument.to_anthropic_schema() for instrument in self._instruments.values()]

    def get_functions_dict(self) -> Dict[str, Callable]:
        """
        Get a dictionary mapping instrument names to their functions.

        This is useful for executing the actual functions when called by an LLM.
        """
        return {name: instrument.function for name, instrument in self._instruments.items()}

    def execute_instrument(self, name: str, **kwargs) -> Any:
        """
        Execute an instrument by name with given parameters.

        Args:
            name: Name of the instrument to execute
            **kwargs: Parameters to pass to the instrument

        Returns:
            Result of the instrument execution
        """
        if name not in self._instruments:
            raise ValueError(f"Instrument '{name}' not found in arsenal '{self.name}'")

        instrument = self._instruments[name]
        logger.info(f"Executing instrument '{name}' from arsenal '{self.name}'")
        return instrument.execute(**kwargs)

    def merge(self, other: 'Arsenal', conflict_strategy: str = "overwrite") -> 'Arsenal':
        """
        Merge another arsenal into this one.

        Args:
            other: Another arsenal to merge
            conflict_strategy: How to handle conflicts ("overwrite", "skip", or "error")

        Returns:
            Self for method chaining
        """
        for name, instrument in other._instruments.items():
            if name in self._instruments:
                if conflict_strategy == "overwrite":
                    logger.info(f"Overwriting instrument '{name}' during merge")
                    self._instruments[name] = instrument
                elif conflict_strategy == "skip":
                    logger.info(f"Skipping duplicate instrument '{name}' during merge")
                    continue
                elif conflict_strategy == "error":
                    raise ValueError(f"Instrument '{name}' already exists in arsenal '{self.name}'")
            else:
                self._instruments[name] = instrument

        logger.info(f"Merged arsenal '{other.name}' into '{self.name}'")
        return self

    def __len__(self) -> int:
        """Return the number of instruments in this arsenal"""
        return len(self._instruments)

    def __contains__(self, name: str) -> bool:
        """Check if an instrument exists in this arsenal"""
        return name in self._instruments

    def __str__(self) -> str:
        return f"Arsenal({self.name}, {len(self._instruments)} instruments)"

    def __repr__(self) -> str:
        return self.__str__()


class ArsenalBuilder:
    """
    Fluent builder for creating Arsenals.

    This provides a clean, chainable interface for building arsenals.
    """

    def __init__(self, name: str, description: str = ""):
        self._arsenal = Arsenal(name, description)

    def with_instrument(self, instrument: Instrument) -> 'ArsenalBuilder':
        """Add an instrument to the arsenal"""
        self._arsenal.add_instrument(instrument)
        return self

    def with_instruments(self, instruments: List[Instrument]) -> 'ArsenalBuilder':
        """Add multiple instruments to the arsenal"""
        for instrument in instruments:
            self._arsenal.add_instrument(instrument)
        return self

    def with_function(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[List] = None
    ) -> 'ArsenalBuilder':
        """Add a function directly as an instrument"""
        instrument = Instrument(name, description, function, parameters)
        self._arsenal.add_instrument(instrument)
        return self

    def build(self) -> Arsenal:
        """Build and return the arsenal"""
        return self._arsenal


# Convenience function for creating arsenals
def create_arsenal(name: str, description: str = "") -> ArsenalBuilder:
    """
    Create a new arsenal using the builder pattern.

    Usage:
        arsenal = create_arsenal("Navigation", "Tools for navigation")\
            .with_instrument(compass)\
            .with_instrument(map_reader)\
            .build()
    """
    return ArsenalBuilder(name, description)
