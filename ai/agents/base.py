from textwrap import dedent
from typing import Generic, List, Optional, TypeVar

from phi.agent import Agent as PhiAgent
from phi.model import Model

from .settings import ComposioAction, agent_settings


class Agent(PhiAgent):
    delegation_directives: Optional[List[str]] = []
    debug_mode: bool = agent_settings.debug_mode
    show_tool_calls: bool = agent_settings.show_tool_calls
    markdown: bool = True
    add_datetime_to_instructions: bool = True


class GPT4Agent(Agent):
    model: Optional[Model] = agent_settings.Models.gpt4


class OllamaAgent(Agent):
    model: Optional[Model] = agent_settings.Models.ollama3_2


class AgentTeam(list, Generic[TypeVar("T", bound=Agent)]):
    def __init__(self, agents: List[Agent] = []):
        # validate all agents during initialization
        self.__check_types(*agents)
        super().__init__(agents)

    def append(self, agent: Agent):
        self.__check_types(agent)
        super().append(agent)

    def extend(self, agents: List[Agent]):
        self.__check_types(*agents)
        super().extend(agents)

    def activate(self, *agents):
        self.extend(agents)

    def __check_types(self, *agents):
        # validate types on extend
        for agent in agents:
            if not isinstance(agent, Agent):
                raise TypeError(
                    f"All items in the extension must be instances of `Agent`, got `{type(agent).__name__}`"
                )

    @property
    def delegation_directives(self) -> str:
        return dedent(
            "\n".join(
                [
                    dir
                    for agent in self
                    for dir in getattr(agent, "delegation_directives", "")
                ]
            )
        ).strip()


__all__ = [
    "GPT4Agent",
    "OllamaAgent",
    "Agent",
    "ComposioAction",
    "AgentTeam",
]
