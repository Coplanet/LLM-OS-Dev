import os
from enum import Enum, unique
from typing import Dict, Optional

from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from composio_agno import Action as ComposioAction
from composio_agno import ComposioToolSet
from pydantic_settings import BaseSettings

from app.auth import User
from helpers.log import logger
from workspace.settings import extra_settings


@unique
class Defaults(Enum):
    TEMPERATURE: float = 0
    MAX_COMPLETION_TOKENS: int = 16000


class AgentSettings(BaseSettings):
    """Agent settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    default_temperature: float = int(os.getenv("DEFAULT_TEMPERATURE", "0"))
    embedding_model: str = os.getenv(
        "OPENAPI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    default_max_completion_tokens: int = int(
        os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", 16000)
    )

    debug_mode: bool = os.getenv("AGENTS_DEBUG_MODE", "false").lower() == "true"
    show_tool_calls: bool = os.getenv("SHOW_TOOL_CALLS", "true").lower() == "true"
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


class AgentConfig:
    user: User
    provider: Optional[str] = None
    model_id: Optional[str] = None
    model_kwargs: Optional[Dict] = {}
    temperature: Optional[float] = None
    enabled: Optional[bool] = None
    max_tokens: Optional[int] = None
    tools: Optional[Dict[Toolkit, Dict]] = {}

    def __init__(
        self,
        user: User,
        provider: str,
        model_id: str,
        model_kwargs: Dict,
        temperature: float,
        enabled: bool,
        max_tokens: int,
        tools: Dict[Toolkit, Dict] = {},
    ):
        self.user = user
        self.provider = provider
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.temperature = temperature
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.tools = tools

    @property
    def is_empty(self):
        return (
            self.provider is None
            and self.model_id is None
            and self.temperature is None
            and self.enabled is None
            and self.max_tokens is None
        )

    @classmethod
    def empty(cls, user: User):
        return cls(user, None, None, None, None, None, None, {})

    def __str__(self):
        return str(
            {
                "model": self.provider,
                "model_id": self.model_id,
                "temperature": self.temperature,
                "enabled": self.enabled,
                "max_tokens": self.max_tokens,
                "tools": [k for k in self.tools.keys()],
                "model_kwargs": self.model_kwargs,
            }
        )

    def __repr__(self):
        return str(self)

    @classmethod
    def default_model(cls):
        from .base import Provider

        return OpenAIChat(
            id="gpt-4o",
            provider=Provider.OpenAI.value,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
            modalities=["text"],
        )

    @property
    def get_model(self):
        from .base import Provider

        if self.is_empty or self.provider is None:
            return self.default_model()

        if self.provider not in Provider:
            logger.warning("Model '%s' is not defined!", self.provider)
            return None

        configs = {}
        model_class = None

        match self.provider:
            case Provider.OpenAI.value:
                model_class = OpenAIChat
                model_id = self.model_id or "gpt-4o"
                configs["api_key"] = extra_settings.gpt_api_key

            case Provider.Groq.value:
                model_class = Groq
                model_id = self.model_id or "llama3-groq-70b-8192-tool-use-preview"
                configs["api_key"] = extra_settings.groq_api_key

            case Provider.Google.value:
                model_class = Gemini
                model_id = self.model_id or "gemini-1.5-flash"
                configs["api_key"] = extra_settings.gemini_api_key

            case Provider.Anthropic.value:
                model_class = Claude
                model_id = self.model_id or "claude-3-5-sonnet-20241022"
                configs["api_key"] = extra_settings.anthropic_api_key

            case _:
                logger.warning("Model '%s' didn't match!", self.provider)
                return None

        if self.max_tokens:
            configs["max_tokens"] = self.max_tokens

        kwargs = {
            "id": model_id,
            "provider": self.provider,
            "temperature": self.temperature,
            **configs,
            **self.model_kwargs,
        }

        if self.provider == Provider.Google.value:
            for key in ["temperature", "max_tokens"]:
                if key in kwargs:
                    del kwargs[key]

        return model_class(**kwargs)


# Create an AgentSettings object
agent_settings = AgentSettings()

__all__ = ["agent_settings", "ComposioAction", "AgentConfig"]
