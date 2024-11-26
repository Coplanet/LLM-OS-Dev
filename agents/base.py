from textwrap import dedent
from typing import Generic, List, Optional, TypeVar

from phi.agent import Agent
from phi.model import Model

from .settings import ComposioAction, agent_settings


class CitexAgent(Agent):
    delegation_directives: Optional[List[str]] = []
    debug_mode: bool = agent_settings.debug_mode
    show_tool_calls: bool = agent_settings.show_tool_calls
    markdown: bool = True
    add_datetime_to_instructions: bool = True


class CitexGPT4Agent(CitexAgent):
    model: Optional[Model] = agent_settings.Models.gpt4


class CitexOllamaAgen(CitexAgent):
    model: Optional[Model] = agent_settings.Models.ollama3_2


class CitextAgentTeam(list, Generic[TypeVar("T", bound=CitexAgent)]):
    def __init__(self, agents: List[CitexAgent]):
        # validate all agents during initialization
        self.__check_types(*agents)
        super().__init__(agents)

    def append(self, agent: CitexAgent):
        self.__check_types(agent)
        super().append(agent)

    def extend(self, agents: List[CitexAgent]):
        self.__check_types(*agents)
        super().extend(agents)

    def activate(self, *agents):
        self.extend(agents)

    def __check_types(self, *agents):
        # validate types on extend
        for agent in agents:
            if not isinstance(agent, CitexAgent):
                raise TypeError(
                    f"All items in the extension must be instances of `CitexAgent`, got `{type(agent).__name__}`"
                )

    @property
    def delegation_directives(self):
        return dedent(
            "\n".join([getattr(agent, "delegation_directives", "") for agent in self])
        ).strip()


__all__ = [
    "CitexGPT4Agent",
    "CitexOllamaAgen",
    "CitexAgent",
    "ComposioAction",
    "CitextAgentTeam",
]
