import json
from typing import List, Optional

from anthropic.types.tool_use_block import ToolUseBlock
from phi.model.message import Message

from helpers.utils import binary2text

from .base import Provider, Transformer


class AnthropicToOpenAI(Transformer):
    def __init__(self):
        super().__init__(Provider.Anthropic, Provider.OpenAI)

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

            elif (
                m.role == "assistant"
                and isinstance(m.content, list)
                and m.content
                and any(
                    [
                        isinstance(c, ToolUseBlock)
                        or (isinstance(c, dict) and c.get("type") == "tool_use")
                        for c in m.content
                    ]
                )
            ):
                meta: Optional[ToolUseBlock] = None
                for c in m.content:
                    if isinstance(c, ToolUseBlock):
                        meta = c
                        break
                    elif isinstance(c, dict) and c.get("type") == "tool_use":
                        meta = ToolUseBlock(
                            type="tool_use",
                            id=c.get("id"),
                            name=c.get("name"),
                            input=c.get("input", "{}"),
                        )
                        break

                if not isinstance(meta, ToolUseBlock):
                    raise ValueError(f"Tool call metadata is not a dictionary: {meta}")

                if not isinstance(meta.input, dict):
                    meta.input = json.loads(meta.input)

                new_messages.append(
                    Message(
                        role="assistant",
                        tool_calls=[
                            {
                                "id": meta.id,
                                "type": "function",
                                "function": {
                                    "name": meta.name,
                                    "arguments": json.dumps(meta.input),
                                },
                            }
                        ],
                    )
                )
                continue
            elif (
                m.role == "user"
                and isinstance(m.content, list)
                and any([c["type"] == "tool_result" for c in m.content])
            ):
                meta = None
                for c in m.content:
                    if c["type"] == "tool_result":
                        meta = c
                        break

                if not isinstance(meta, dict):
                    raise ValueError(
                        f"Tool result metadata is not a dictionary: {meta}"
                    )

                m.role = "tool"
                m.tool_call_id = meta["tool_use_id"]
                m.content = meta["content"]

            if isinstance(m.content, list) and m.content:
                for index in range(len(m.content)):
                    c = m.content[index]
                    if isinstance(c, dict):
                        if c.get("type") == "image":
                            image_media_type = c["source"]["media_type"]
                            img = c["source"]["data"]
                            if isinstance(img, bytes):
                                img = binary2text(img, image_media_type)
                            else:
                                img = str(img)

                            if img.startswith("http"):
                                image_data = img
                            else:
                                image_data = "data:{};base64,{}".format(
                                    image_media_type, img
                                )

                            m.content[index] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data,
                                },
                            }

            new_messages.append(m)

        agent.prune_openai_messages(new_messages)

        return new_messages
