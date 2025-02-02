from typing import List

from agno.models.message import Message

from .anthropic_to_openai import AnthropicToOpenAI
from .base import Provider, Transformer
from .openai_to_groq import OpenAIToGroq


class AnthropicToGroq(Transformer):
    def __init__(self):
        super().__init__(Provider.Anthropic, Provider.Groq)

    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        return OpenAIToGroq().transform_messages(
            agent,
            system_message,
            AnthropicToOpenAI().transform_messages(agent, system_message, messages),
        )
