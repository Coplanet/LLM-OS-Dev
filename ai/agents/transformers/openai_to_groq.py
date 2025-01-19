from typing import List

from phi.model.message import Message

from .base import Provider, Transformer


class OpenAIToGroq(Transformer):
    def __init__(self):
        super().__init__(Provider.OpenAI, Provider.Groq)

    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        new_messages = []
        for m in messages:
            if m.role == "developer":
                m.role = "system"

            new_messages.append(m)

        return new_messages
