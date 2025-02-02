from typing import List

from agno.models.message import Message

from .base import Provider, Transformer
from .groq_to_openai import GroqToOpenAI
from .openai_to_anthropic import OpenAIToAnthropic


class GroqToAnthropic(Transformer):
    def __init__(self):
        super().__init__(Provider.Groq, Provider.Anthropic)

    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        return OpenAIToAnthropic().transform_messages(
            agent,
            system_message,
            GroqToOpenAI().transform_messages(agent, system_message, messages),
        )
