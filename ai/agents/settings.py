import os
from enum import Enum, unique
from typing import Optional

from composio_phidata import Action as ComposioAction
from composio_phidata import ComposioToolSet
from phi.model.openai import OpenAIChat
from pydantic_settings import BaseSettings

from workspace.settings import extra_settings


@unique
class Defaults(Enum):
    TEMPERATURE: float = 0
    MAX_COMPLETION_TOKENS: int = 16000


class AgentSettings(BaseSettings):
    """Agent settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    gpt_4: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    default_max_completion_tokens: int = Defaults.MAX_COMPLETION_TOKENS.value
    default_temperature: float = Defaults.TEMPERATURE.value

    debug_mode: bool = False
    show_tool_calls: bool = True
    composio_tools: Optional[ComposioToolSet] = None

    def __init__(self, *args, **kwargs):
        # pass arguments to the parent constructor
        super().__init__(*args, **kwargs)

        # initialize composio_tools if not already set
        self.composio_tools = self.composio_tools or ComposioToolSet()

        # update debug_mode and show_tool_calls using helper methods
        self.debug_mode = self._get_boolean_env("AGENTS_DEBUG_MODE", self.debug_mode)
        self.show_tool_calls = self._get_boolean_env(
            "SHOW_TOOL_CALLS", self.show_tool_calls
        )

    def _getenv(self, key, default: Optional[str] = None) -> str:
        return os.getenv(key, default)

    def _get_boolean_env(self, key, default: bool = False) -> bool:
        return self._getenv(key, "true" if default else "false").lower() == "true"

    class Models:
        @classmethod
        def get_gpt_model(cls, model_id: str):
            return OpenAIChat(
                id=model_id,
                max_tokens=Defaults.MAX_COMPLETION_TOKENS.value,
                temperature=Defaults.TEMPERATURE.value,
                api_key=extra_settings.gpt_api_key,
            )


# Create an AgentSettings object
agent_settings = AgentSettings()

__all__ = ["agent_settings", "ComposioAction", "Defaults"]
