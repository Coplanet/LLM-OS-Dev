from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import List

from agno.models.message import Message


@unique
class Provider(Enum):
    OpenAI = "OpenAI"
    Groq = "Groq"
    Google = "Google"
    Anthropic = "Anthropic"

    @classmethod
    def __contains__(cls, item):
        return item in cls._value2member_map_


class Transformer(ABC):
    def __init__(self, from_provider: Provider, to_provider: Provider):
        self.from_provider = from_provider
        self.to_provider = to_provider

    def transform(self, agent) -> "Transformer":
        from ..base import Agent

        if not isinstance(agent, Agent):
            raise ValueError("Agent must be an instance of Agent")

        SYSTEM_MESSAGE: Message = agent.get_system_message()

        for prev_run in agent.memory.runs:
            if prev_run.response and prev_run.response.messages:
                prev_run.response.messages = self.transform_messages(
                    agent, SYSTEM_MESSAGE, prev_run.response.messages
                )

        agent.memory.messages = self.transform_messages(
            agent, SYSTEM_MESSAGE, agent.memory.messages
        )

    @abstractmethod
    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        pass
