from textwrap import dedent
from typing import Any, Dict, Generic, List, Optional, TypeVar

from phi.agent import Agent as PhiAgent
from phi.model import Model
from phi.model.openai import OpenAIChat

from .settings import ComposioAction, Defaults, agent_settings, extra_settings

DEFAULT_GPT_MODEL_CONFIG = {
    "max_tokens": Defaults.MAX_COMPLETION_TOKENS.value,
    "temperature": Defaults.TEMPERATURE.value,
    "api_key": extra_settings.gpt_api_key,
}


class Agent(PhiAgent):
    enabled: bool = True
    model: Optional[Model] = OpenAIChat(id="gpt-4o")

    delegation_directives: Optional[List[str]] = []
    debug_mode: bool = agent_settings.debug_mode
    show_tool_calls: bool = agent_settings.show_tool_calls
    markdown: bool = True
    add_datetime_to_instructions: bool = True

    def register_or_load(
        self,
        default_model_config: Dict[str, Any] = DEFAULT_GPT_MODEL_CONFIG,
        default_agent_config: Dict[str, Any] = {},
    ):
        from dashboard.models import AgentConfig

        for field, config in default_model_config.items():
            setattr(self.model, field, config)

        for field, config in default_agent_config.items():
            setattr(self, field, config)

        return AgentConfig.register_or_load(
            self, default_model_config, default_agent_config
        )


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
    "Agent",
    "ComposioAction",
    "AgentTeam",
]
