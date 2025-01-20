import hashlib
import imghdr
from textwrap import dedent
from typing import Generic, List, Optional, Tuple, TypeVar, Union

from phi.agent import Agent as PhiAgent
from phi.agent.session import AgentSession
from phi.model.anthropic import Claude
from phi.model.base import Model
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.message import Message
from phi.model.openai import OpenAIChat

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

    def read_from_storage(self) -> Optional[AgentSession]:
        session = super().read_from_storage()

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

    def write_to_storage(self) -> Optional[AgentSession]:
        type_mapping = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }

        def normalize_images(messages: List[Message]):
            for m in messages:
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
        from app.components.popup import MODELS, SupportStrength

        return {
            type_: strength
            for type_, strength in MODELS[self.model.provider][self.model.id]
            .get("supports", {})
            .items()
            if strength != SupportStrength.NotSupported
        }

    @property
    def limits(self) -> dict:
        from app.components.popup import MODELS

        return MODELS[self.model.provider][self.model.id].get("limits", {})

    def prune_messages(self, messages: List[Message]):
        if self.model.provider == Provider.Anthropic.value:
            self.prune_anthropic_messages(messages)
        elif self.model.provider == Provider.OpenAI.value:
            self.prune_openai_messages(messages)

    def prune_openai_messages(self, messages: List[Message]):
        # remove inconsistent tool calls/results
        index2remove = []
        for index in range(len(messages)):
            if index == 0:
                continue
            cp = messages[index]
            pp = messages[index - 1]
            np = messages[index + 1] if index + 1 < len(messages) else None
            if cp.role == "tool" and not pp.tool_calls:
                index2remove.append(index)
            if np and cp.role == "assistant" and not np.role == "tool":
                index2remove.append(index)

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
                if (
                    pp.role != "assistant"
                    or not isinstance(pp.content, list)
                    or not any(c.get("type") == "tool_use" for c in pp.content)
                ):
                    index2remove.append(index)
            # remove tool uses that are not followed by a tool result
            if (
                np
                and cp.role == "assistant"
                and isinstance(cp.content, list)
                and any(c.get("type") == "tool_use" for c in cp.content)
            ):
                if (
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
