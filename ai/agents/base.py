import hashlib
import imghdr
from textwrap import dedent
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from uuid import uuid4

import backoff
import requests
from agno.agent import Agent as PhiAgent
from agno.agent import AgentMemory, AgentSession, RunResponse
from agno.models.anthropic import Claude
from agno.models.base import Model
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from anthropic.types import ToolUseBlock
from composio.exceptions import ComposioSDKError
from composio_agno import App, ComposioToolSet
from pydantic import ConfigDict, Field

from app.auth import User
from db.session import get_db_context
from db.tables import UserIntegration, UserNextOp
from helpers.log import logger
from helpers.utils import binary2text, text2binary

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    transformers: dict[str, Transformer] = {}
    model: Optional[Model] = OpenAIChat(id="gpt-4o")

    agent_config: AgentConfig = Field(default=None)
    available_models: dict[str, dict[str, Any]] = Field(default={})

    delegation_directives: Optional[List[str]] = []
    debug_mode: bool = agent_settings.debug_mode
    show_tool_calls: bool = agent_settings.show_tool_calls
    markdown: bool = True
    add_datetime_to_instructions: bool = True

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        available_models: dict[str, dict[str, Any]] = {},
        delegation_directives: Optional[List[str]] = [],
        *args,
        **kwargs,
    ):
        self.run_id
        from app.components.model_config import MODELS

        if agent_config:
            if not isinstance(agent_config, AgentConfig):
                raise Exception("agent_config must be an instance of AgentConfig")
            kwargs["model"] = agent_config.get_model

        if available_models and not isinstance(available_models, dict):
            raise Exception("available_models must be a dictionary")

        if "model" not in kwargs or kwargs["model"] is None:
            kwargs["model"] = AgentConfig.default_model()

        super().__init__(*args, **kwargs)

        self.delegation_directives = delegation_directives

        self.agent_config = agent_config
        self.available_models = (
            available_models or MODELS[self.model.provider][self.model.id]
        )

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

    def read_from_storage(self) -> Optional[AgentSession]:
        session = super().read_from_storage()
        return session

        if self.model.provider == Provider.Anthropic.value:

            def normalize_images(messages: List[Message]):
                for m in messages:
                    if m.images:
                        for index in range(len(m.images)):
                            img = m.images[index]
                            if not isinstance(img, bytes):
                                m.images[index] = text2binary(img)

            for prev_run in self.memory.runs:
                if prev_run.response and prev_run.response.messages:
                    normalize_images(prev_run.response.messages)

            normalize_images(self.memory.messages)

        return session

    def generate_session_name(self) -> str:
        """Generate a name for the session using the last 6 messages from the memory"""

        if self.model is None:
            raise Exception("Model not set")

        gen_session_name_prompt = "Conversation\n"
        messages_for_generating_session_name = []
        self.memory = cast(AgentMemory, self.memory)
        try:
            message_pairs = self.memory.get_message_pairs()
            for message_pair in message_pairs[-3:]:
                messages_for_generating_session_name.append(message_pair[0])
                messages_for_generating_session_name.append(message_pair[1])
        except Exception as e:
            logger.warning(f"Failed to generate name: {e}")

        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f"{message.role.upper()}: {message.content}\n"

        gen_session_name_prompt += "\n\nConversation Name: "

        system_message = Message(
            role=self.get_system_message_role(),
            content="Please provide a suitable name for this conversation in maximum 5 words. "
            "Remember, do not exceed 5 words.",
        )
        user_message = Message(
            role=self.user_message_role, content=gen_session_name_prompt
        )
        generate_name_messages = [system_message, user_message]
        generated_name = self.model.response(messages=generate_name_messages)
        content = generated_name.content
        if content is None:
            logger.error("Generated name is None. Trying again.")
            return self.generate_session_name()
        if len(content.split()) > 15:
            logger.error("Generated name is too long. Trying again.")
            return self.generate_session_name()
        return content.replace('"', "").strip()

    def write_to_storage(self) -> Optional[AgentSession]:
        return super().write_to_storage()

        type_mapping = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }

        def normalize_images(messages: List[Message]):
            for m in messages:
                if not hasattr(m, "images"):
                    continue

                if m.images:
                    for index in range(len(m.images)):
                        img = m.images[index]
                        if isinstance(img, bytes):
                            image_type = type_mapping[
                                imghdr.what(None, h=img) or "webp"
                            ]
                            m.images[index] = binary2text(img, image_type)

                if isinstance(m.content, list) and len(m.content) > 1:
                    hash2index = {}
                    index2remove = []
                    for index in range(len(m.content)):
                        c = m.content[index]
                        if isinstance(c, dict):
                            data = None
                            if c.get("type") == "image":
                                data = c["source"]["data"]
                            elif c.get("type") == "image_url":
                                data = c["image_url"]["url"]

                            if data:
                                if not isinstance(data, bytes):
                                    data = str(data).encode()

                                KEY = hashlib.md5(data).hexdigest()
                                if KEY not in hash2index:
                                    hash2index[KEY] = index
                                else:
                                    index2remove.append(index)
                    if index2remove:
                        index2remove.reverse()
                        for index in index2remove:
                            m.content.pop(index)

        for prev_run in self.memory.runs:
            if prev_run.response and prev_run.response.messages:
                normalize_images(prev_run.response.messages)

        normalize_images(self.memory.messages)

        return super().write_to_storage()

    @property
    def supports(self) -> dict:
        from app.models import SupportStrength

        return {
            type_: strength
            for type_, strength in self.available_models.get("supports", {}).items()
            if strength != SupportStrength.NotSupported
        }

    @property
    def limits(self) -> dict:
        return self.available_models.get("limits", {})

    def prune_messages(self, messages: List[Message]):
        if self.model.provider == Provider.Anthropic.value:
            self.prune_anthropic_messages(messages)
        elif self.model.provider == Provider.OpenAI.value:
            self.prune_openai_messages(messages)

    def prune_openai_messages(self, messages: List[Message]):
        # remove inconsistent tool calls/results
        index2remove = []
        index = 0

        while index < len(messages):
            cp = messages[index]
            if cp.role == "assistant" and cp.tool_calls:
                if isinstance(cp.tool_calls, list):
                    cindex = index
                    tools_index = []
                    while index < len(messages):
                        index += 1
                        wp = messages[index]
                        if wp.role == "tool":
                            tools_index.append(index)
                        else:
                            break

                    if len(cp.tool_calls) != len(tools_index):
                        index2remove.append(cindex)
                        index2remove.extend(tools_index)

            index += 1

        index2remove.reverse()
        for index in index2remove:
            messages.pop(index)

    def prune_anthropic_messages(self, messages: List[Message]):
        # remove inconsistent tool calls/results
        index2remove = []
        for index in range(len(messages)):
            if index == 0:
                continue
            cp = messages[index]
            pp = messages[index - 1]
            np = messages[index + 1] if index + 1 < len(messages) else None
            # remove tool results that are not preceded by a tool use
            if (
                cp.role == "user"
                and isinstance(cp.content, list)
                and any(c.get("type") == "tool_result" for c in cp.content)
            ):
                if pp.role != "assistant" or not isinstance(pp.content, list):
                    TOOLUSE = False
                    for c in pp.content:
                        if (
                            isinstance(c, dict) and c.get("type") == "tool_use"
                        ) or isinstance(c, ToolUseBlock):
                            TOOLUSE = True
                            break
                    if not TOOLUSE:
                        index2remove.append(index)
            # remove tool uses that are not followed by a tool result
            if np and cp.role == "assistant" and isinstance(cp.content, list):
                TOOLUSE = False
                for c in cp.content:
                    if (
                        isinstance(c, dict) and c.get("type") == "tool_use"
                    ) or isinstance(c, ToolUseBlock):
                        TOOLUSE = True
                        break

                if TOOLUSE and (
                    np.role != "user"
                    or not isinstance(np.content, list)
                    or not any(c.get("type") == "tool_result" for c in np.content)
                ):
                    index2remove.append(index)

        index2remove.reverse()
        for index in index2remove:
            messages.pop(index)

    def get_messages_for_run(
        self, *args, **kwargs
    ) -> Tuple[Optional[Message], List[Message], List[Message]]:
        # 3. Prepare messages for this run
        system_message, user_messages, messages_for_model = (
            super().get_messages_for_run(*args, **kwargs)
        )

        from app.models import SupportTypes

        # Groq does not support multiple images in the messages
        if self.model.provider == Provider.Groq.value:

            IMAGE_IN_LIMITS = self.limits.get(SupportTypes.ImageIn, 0)

            if SupportTypes.ImageIn in self.supports and IMAGE_IN_LIMITS > 0:
                messages_with_images = []

                def normalize_messages(messages: List[Message]):
                    for index, message in enumerate(messages):
                        if message.images:
                            messages_with_images.append(
                                {"index": index, "container": messages}
                            )

                        if self.model.id == "llama-3.2-90b-vision-preview":
                            if (
                                isinstance(message.content, list)
                                and len(message.content) > 1
                            ):
                                content = None
                                for c in message.content:
                                    if c.get("type") == "text":
                                        content = c["text"]
                                        break

                                if content:
                                    message.content = content

                normalize_messages(messages_for_model)
                normalize_messages(user_messages)

                for index in messages_with_images[:-IMAGE_IN_LIMITS]:
                    index["container"][index["index"]].images = None

        if SupportTypes.ImageIn not in self.supports:
            # Remove images from messages we send to the model
            for message in messages_for_model:
                if message.images:
                    message.images = None

            for message in user_messages:
                if message.images:
                    message.images = None

        self.prune_messages(messages_for_model)

        if self.model.provider == Provider.OpenAI.value:
            if self.model.id in ["o1-mini"]:
                index2remove = []
                for index, message in enumerate(messages_for_model):
                    if message.role not in ["user", "assistant"]:
                        index2remove.append(index)

                index2remove.reverse()
                for index in index2remove:
                    messages_for_model.pop(index)

        return system_message, user_messages, messages_for_model

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

    def transformer_exists(self, to_provider: Provider) -> bool:
        for transformer in self.transformers.values():
            if (
                transformer.from_provider.value == self.model.provider
                and transformer.to_provider.value == to_provider.value
            ):
                return True
        return False

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

    @classmethod
    def get_tools_as_composio_tools(
        cls, agent_name: str, config: AgentConfig, app: App
    ) -> List[Toolkit]:
        user_integrations = UserIntegration.get_integrations(
            config.user.user_id, app=str(app)
        )

        toolset = None

        if user_integrations.count() > 0:
            try:
                toolset = ComposioToolSet(
                    entity_id=config.user.user_id,
                    connected_account_ids={
                        i.app: i.connection_id for i in user_integrations
                    },
                )

            except ComposioSDKError as e:
                logger.error(f"Error getting tools as composio tools: {e}")
                for i in user_integrations:
                    if i.app in str(e):
                        i.delete()

        config.tools = []
        available_tools = []

        if toolset:
            from ai.coordinators.composio_tools import COMPOSIO_ACTIONS

            for integration in user_integrations:
                if integration.app in COMPOSIO_ACTIONS:
                    details = COMPOSIO_ACTIONS[integration.app]
                    for order, instance in enumerate(
                        toolset.get_tools(actions=details["actions"])
                    ):
                        name = details["name"]
                        available_tools.append(
                            {
                                "group": name,
                                "order": 500 + order + 1,
                                "instance": instance,
                                "name": instance.name,
                                "icon": details["icon"],
                            }
                        )

        if not available_tools:
            return []

        config.tools = available_tools

        from helpers.tool_processor import process_tools

        tools, _ = process_tools(agent_name, config, available_tools)

        return tools


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


class ComposioAgent(Agent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    app: Any = Field(default=None)
    user: User = Field(default=None)
    app_name: str = Field(default=None)

    def __init__(self, app: App, user: User, *args, **kwargs):
        from ai.coordinators.composio_tools import COMPOSIO_ACTIONS

        if app not in COMPOSIO_ACTIONS:
            raise ValueError(f"App {app} is not supported")

        kwargs["instructions"] = [
            (
                "<CRITICAL INSTRUCTIONS>\n"
                "  1. {}\n"
                "  2. {}\n"
                "</CRITICAL INSTRUCTIONS>".format(
                    "ALWAYS USE YOUR TOOLS ENABLED FOR YOU, NEVER DO ANYTHING EXCEPT USING THE TOOLS ENABLED FOR YOU.",
                    (
                        "YOU MAY ONLY USE A FUNCTION, DO NOT MAKE UP AN ANSWER. EVEN IF THE TOOL WAS CALLED BEFORE, "
                        "YOU HAVE TO CALL IT AGAIN.\n"
                    ),
                )
            )
        ]

        super().__init__(*args, **kwargs)

        self.app = app
        self.user = user
        self.app_name = str(app)

        @backoff.on_exception(
            backoff.expo,
            (requests.exceptions.RequestException),
            max_time=60,
            max_tries=10,
        )
        def run_with_backoff(self: "ComposioAgent", kwargs: dict):
            if self.has_integrations:
                self.tools = self.get_tools_as_composio_tools(
                    self.name, kwargs.get("agent_config", None), self.app
                )
            else:
                self.tools = ComposioToolSet().get_tools(
                    actions=COMPOSIO_ACTIONS[self.app]["actions"]
                )

        run_with_backoff(self, kwargs)

    @property
    def has_integrations(self):
        return (
            UserIntegration.get_integrations(self.user.user_id, self.app_name).count()
            > 0
        )

    def run(
        self, message: Optional[Union[str, List, Dict, Message]] = None, *args, **kwargs
    ) -> Union[RunResponse, Iterator[RunResponse]]:
        if not self.has_integrations:
            with get_db_context() as db:
                UserNextOp.save_op(
                    db,
                    self.user.session_id,
                    UserNextOp.AUTH_USER,
                    {"app": self.app_name},
                )
                # Create the run_response object
                self.run_id = str(uuid4())
                self.run_response = RunResponse(
                    run_id=self.run_id,
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    content=dedent(
                        """
                        ANNOUNCE THIS AS RESPONSE:
                        The `{}` operation was not successful. Ask user to integrate his/her `{}` account with you.
                        """
                    )
                    .strip()
                    .format(self.app_name, self.app_name),
                )
                self.stream = kwargs.get("stream", False) and self.is_streamable
                if self.stream and self.is_streamable:
                    yield self.run_response
                else:
                    return self.run_response

        kwargs["stream"] = False
        return super().run(message, *args, **kwargs)


__all__ = [
    "Agent",
    "ComposioAgent",
    "ComposioAction",
    "AgentTeam",
]
