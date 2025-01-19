from textwrap import dedent
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from phi.agent import Agent as PhiAgent
from phi.model.anthropic import Claude
from phi.model.base import Model
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.message import Message
from phi.model.openai import OpenAIChat

from helpers.log import logger

from .settings import AgentConfig, ComposioAction, agent_settings, extra_settings

DEFAULT_GPT_MODEL_CONFIG = {
    "max_tokens": agent_settings.default_max_completion_tokens,
    "temperature": agent_settings.default_temperature,
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

    @property
    def model_type(self):
        if isinstance(self.model, OpenAIChat):
            return "OpenAI"
        if isinstance(self.model, Groq):
            return "Groq"
        if isinstance(self.model, Gemini):
            return "Google"
        if isinstance(self.model, Claude):
            return "Anthropic"
        logger.warning(f"Model type '{self.model}' is not defined!")
        return None

    @property
    def label(self):
        from app.utils import to_label

        return to_label(self.name)

    def get_messages_for_run(
        self,
        *,
        message: Optional[Union[str, List, Dict, Message]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Message], List[Message], List[Message]]:
        """This function returns:
            - the system message
            - a list of user messages
            - a list of messages to send to the model

        To build the messages sent to the model:
        1. Add the system message to the messages list
        2. Add extra messages to the messages list if provided
        3. Add history to the messages list
        4. Add the user messages to the messages list

        Returns:
            Tuple[Message, List[Message], List[Message]]:
                - Optional[Message]: the system message
                - List[Message]: user messages
                - List[Message]: messages to send to the model
        """
        (
            system_message,
            user_messages,
            messages_for_model,
        ) = super().get_messages_for_run(
            message=message,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            **kwargs,
        )

        return system_message, user_messages, messages_for_model


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
