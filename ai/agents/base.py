from textwrap import dedent
from typing import Generic, List, Optional, TypeVar, Union

from phi.agent import Agent as PhiAgent
from phi.model.anthropic import Claude
from phi.model.base import Model
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat

from helpers.log import logger

from .settings import AgentConfig, ComposioAction, agent_settings, extra_settings
from .transformers import (
    AnthropicToGroq,
    AnthropicToOpenAI,
    GroqToAnthropic,
    GroqToOpenAI,
    OpenAIToAnthropic,
    OpenAIToGroq,
    Provider,
    Transformer,
)

DEFAULT_GPT_MODEL_CONFIG = {
    "max_tokens": agent_settings.default_max_completion_tokens,
    "temperature": agent_settings.default_temperature,
    "api_key": extra_settings.gpt_api_key,
}


class Agent(PhiAgent):
    enabled: bool = True
    transformers: dict[str, Transformer] = {}
    model: Optional[Model] = OpenAIChat(id="gpt-4o")

    delegation_directives: Optional[List[str]] = []
    debug_mode: bool = agent_settings.debug_mode
    show_tool_calls: bool = agent_settings.show_tool_calls
    markdown: bool = True
    add_datetime_to_instructions: bool = True

    def __init__(self, *args, **kwargs):
        agent_config: AgentConfig = kwargs.pop("agent_config", None)
        if agent_config:
            if not isinstance(agent_config, AgentConfig):
                raise Exception("agent_config must be an instance of AgentConfig")
            kwargs["model"] = agent_config.get_model

        if "model" not in kwargs or kwargs["model"] is None:
            kwargs["model"] = AgentConfig.default_model()

        super().__init__(*args, **kwargs)

        logger.debug(
            "Agent '%s' initialized using model: '%s' with temperature: '%s'",
            self.name or "n/a",
            self.model.id,
            str(getattr(self.model, "temperature", "n/a")),
        )

        self.register_transformer(GroqToOpenAI())
        self.register_transformer(GroqToAnthropic())
        self.register_transformer(AnthropicToGroq())
        self.register_transformer(AnthropicToOpenAI())
        self.register_transformer(OpenAIToGroq())
        self.register_transformer(OpenAIToAnthropic())

    @property
    def model_type(self):
        if isinstance(self.model, OpenAIChat):
            return Provider.OpenAI.value
        if isinstance(self.model, Groq):
            return Provider.Groq.value
        if isinstance(self.model, Gemini):
            return Provider.Google.value
        if isinstance(self.model, Claude):
            return Provider.Anthropic.value
        logger.warning(f"Model type '{self.model}' is not defined!")
        return None

    @property
    def label(self):
        from app.utils import to_label

        return to_label(self.name)

    def _transformation_key(
        self, from_provider: Union[Provider, str], to_provider: Union[Provider, str]
    ):
        if isinstance(from_provider, Provider):
            from_provider = from_provider.value
        if isinstance(to_provider, Provider):
            to_provider = to_provider.value
        return f"{from_provider}:{to_provider}"

    def register_transformer(self, transformer: Transformer) -> "Agent":
        self.transformers[
            self._transformation_key(transformer.from_provider, transformer.to_provider)
        ] = transformer
        return self

    def transform(self, to_provider: Provider) -> "Agent":
        if self.model.provider == to_provider.value:
            return self

        TRANSFORMATION_KEY = self._transformation_key(self.model.provider, to_provider)

        if TRANSFORMATION_KEY in self.transformers:
            self.read_from_storage()

            self.transformers[TRANSFORMATION_KEY].transform(self)

            self.write_to_storage()

        else:
            logger.warning(
                f"No transformer found for {self.model.provider} to {to_provider}"
            )
        return self


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
        DELIMITER = "\n---\n"

        return (
            DELIMITER
            + dedent(
                DELIMITER.join(
                    [
                        dir.strip()
                        for agent in self
                        for dir in getattr(agent, "delegation_directives", "")
                    ]
                )
            ).strip()
        )


__all__ = [
    "Agent",
    "ComposioAction",
    "AgentTeam",
]
