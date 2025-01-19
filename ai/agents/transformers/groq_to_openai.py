from typing import List

from phi.model.message import Message

from .base import Provider, Transformer


class GroqToOpenAI(Transformer):
    def __init__(self):
        super().__init__(Provider.Groq, Provider.OpenAI)

    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        new_messages = []
        for m in messages:
            if m.role == "assistant" and (
                (system_message and m.content == system_message.content)
                or (agent.introduction and m.content == agent.introduction)
            ):
                m.role = "system"

            new_messages.append(m)

        return new_messages
