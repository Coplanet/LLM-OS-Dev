from textwrap import dedent
from typing import Any, Dict, Generic, List, Optional, TypeVar

from phi.agent import Agent as PhiAgent
from phi.model import Model
from phi.model.groq import Groq
from phi.model.ollama import Ollama
from phi.model.openai import OpenAIChat
from phi.utils.log import logger

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
        force_model: Model = None,
    ):
        from dashboard.models import AgentConfig

        for field, config in default_model_config.items():
            setattr(self.model, field, config)

        for field, config in default_agent_config.items():
            setattr(self, field, config)

        return AgentConfig.register_or_load(
            self, default_model_config, default_agent_config, force_model
        )

    @property
    def model_type(self):
        if isinstance(self.model, OpenAIChat):
            return "GPT"
        if isinstance(self.model, Groq):
            return "Groq"
        if isinstance(self.model, Ollama):
            return "LLaMA"
        logger.warning(f"Model type '{self.model}' is not defined!")
        return None

    @property
    def label(self):
        return self.name.lower().replace(" ", "_")

    @staticmethod
    def get_model(model: str = None, model_id: str = None, templature: float = 0):
        models = {"GPT", "Groq", "LLaMA"}
        if model is None:
            return None

        if model not in models:
            logger.warning(f"Model '{model}' is not defined!")
            return None

        configs = {}
        model_class = None

        match model:
            case "GPT":
                model_class = OpenAIChat
                model_id = model_id or "gpt-4o"
                configs["api_key"] = extra_settings.gpt_api_key

            case "Groq":
                model_class = Groq
                model_id = model_id or "llama3-groq-70b-8192-tool-use-preview"
                configs["api_key"] = extra_settings.groq_api_key
            case "LLaMA":
                model_class = Ollama
                model_id = model_id or "llama3.2"
                configs["host"] = extra_settings.ollama_host

            case _:
                logger.warning(f"Model '{model}' didn't match!")
                return None

        return model_class(
            id=model_id,
            temperature=templature,
            max_tokens=agent_settings.default_max_completion_tokens,
            **configs,
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
