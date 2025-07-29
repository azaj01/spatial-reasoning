from abc import ABC, abstractmethod

from ..agents import BaseAgent


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, agent: BaseAgent, **kwargs):
        self.agent = agent
        self.kwargs = kwargs

    @abstractmethod
    def execute(self, **kwargs):
        """ Run task executor """
        pass
