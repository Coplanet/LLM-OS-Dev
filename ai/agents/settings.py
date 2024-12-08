import os
from enum import Enum, unique
from typing import Optional

from composio_phidata import Action as ComposioAction
from composio_phidata import ComposioToolSet
from phi.model.groq import Groq
from phi.model.ollama import Ollama
from phi.model.openai import OpenAIChat
from phi.utils.log import logger
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

    default_temperature: float = int(os.getenv("DEFAULT_TEMPERATURE", "0"))
    embedding_model: str = os.getenv(
        "OPENAPI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    default_max_completion_tokens: int = int(
        os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", 16000)
    )

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


class AgentConfig:
    provider: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    enabled: Optional[bool] = None
    max_tokens: Optional[int] = None

    def __init__(
        self,
        provider: str,
        model_id: str,
        temperature: float,
        enabled: bool,
        max_tokens: int,
    ):
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.enabled = enabled
        self.max_tokens = max_tokens

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
    def empty(cls):
        return cls(None, None, None, None, None)

    def __str__(self):
        return str(
            {
                "model": self.provider,
                "model_id": self.model_id,
                "temperature": self.temperature,
                "enabled": self.enabled,
                "max_tokens": self.max_tokens,
            }
        )

    def __repr__(self):
        return str(self)

    @classmethod
    def default_model(cls):
        return OpenAIChat(
            id="gpt-4o",
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        )

    @property
    def get_model(self):
        models = {"OpenAI", "Groq", "Ollama"}
        if self.is_empty or self.provider is None:
            return self.default_model()

        if self.provider not in models:
            logger.warning("Model '%s' is not defined!", self.provider)
            return None

        if not self.max_tokens:
            self.max_tokens = agent_settings.default_max_completion_tokens

        configs = {}
        model_class = None

        match self.provider:
            case "OpenAI":
                model_class = OpenAIChat
                model_id = self.model_id or "gpt-4o"
                configs["api_key"] = extra_settings.gpt_api_key

            case "Groq":
                model_class = Groq
                model_id = self.model_id or "llama3-groq-70b-8192-tool-use-preview"
                configs["api_key"] = extra_settings.groq_api_key
            case "Ollama":
                model_class = Ollama
                model_id = self.model_id or "llama3.2"
                configs["host"] = extra_settings.ollama_host

            case _:
                logger.warning("Model '%s' didn't match!", self.provider)
                return None

        return model_class(
            id=model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **configs,
        )


# Create an AgentSettings object
agent_settings = AgentSettings()

__all__ = ["agent_settings", "ComposioAction", "AgentConfig"]
