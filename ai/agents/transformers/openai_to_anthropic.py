import json
from typing import List

from anthropic.types.tool_use_block import ToolUseBlock
from phi.model.message import Message

from .base import Provider, Transformer


class OpenAIToAnthropic(Transformer):
    def __init__(self):
        super().__init__(Provider.OpenAI, Provider.Anthropic)

    def transform_messages(
        self, agent, system_message: Message, messages: List[Message]
    ) -> List[Message]:
        new_messages = []
        for m in messages:
            if m.role == "developer":
                m.role = "system"

            elif "tool_calls" in m.model_fields_set and isinstance(m.tool_calls, list):
                meta = m.tool_calls[0]
                if not isinstance(meta, dict):
                    raise ValueError(f"Tool call metadata is not a dictionary: {meta}")

                tooluse = ToolUseBlock(
                    type="tool_use",
                    id=meta.get("id"),
                    name=meta.get("function", {}).get("name"),
                    input=meta.get("function", {}).get("arguments", "{}"),
                )
                if not isinstance(tooluse.input, dict):
                    tooluse.input = json.loads(tooluse.input)

                new_messages.append(Message(role="assistant", content=[tooluse]))
                continue

            elif m.role == "tool":
                m.role = "user"
                m.content = [
                    {
                        "type": "tool_result",
                        "content": m.content,
                        "tool_use_id": m.tool_call_id,
                    }
                ]
            new_messages.append(m)

        return new_messages
