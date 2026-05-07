"""Base class for GREEN opponent policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from flyer.environment.environment import FlyerAction


class IPolicy(ABC):
    """Abstract base class for GREEN flyer behavior policies."""

    @abstractmethod
    def get_action(self) -> FlyerAction:
        """Return action for GREEN flyer."""
        pass
