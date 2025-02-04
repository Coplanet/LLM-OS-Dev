from collections import OrderedDict
from enum import Enum, unique

from ai.agents.base import Provider
from ai.agents.settings import agent_settings

DEFAULT_TEMPERATURE = OrderedDict(
    [
        ("Deterministic", {"value": 0.0, "icon": ":material/border_outer:"}),
        ("Balanced", {"value": 0.75, "icon": ":material/balance:"}),
        ("Creative", {"value": 1.5, "icon": ":material/emoji_objects:"}),
    ]
)


@unique
class SupportStrength(Enum):
    NotSupported = "None"
    Limited = "Limited"
    Partially = "Partially"
    Weak = "Weak"
    Acceptable = "Acceptable"
    Full = "Full"

    def color(self):
        return {
            SupportStrength.NotSupported: "red",
            SupportStrength.Partially: "orange",
            SupportStrength.Weak: "yellow",
            SupportStrength.Acceptable: "forestgreen",
            SupportStrength.Full: "lime",
        }[self]


@unique
class SupportTypes(Enum):
    TextIn = "Text In"
    TextOut = "Text Out"
    ImageIn = "Image In"
    # AudioIn = "Audio In"
    # AudioOut = "Audio Out"
    ImageOut = "Image Out"
    FunctionCalling = "Function Calling"
    Reasoning = "Reasoning"
    ParallelToolCalling = "Parallel Tool Calling"

    @classmethod
    def __contains__(cls, item):
        return item in cls._value2member_map_


MODELS = {
    Provider.OpenAI.value: {
        "o3-mini": {
            "kwargs": {
                "max_completion_tokens": 100_000,
            },
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Full,
            },
        },
        "gpt-4o": {
            "max_token_size": agent_settings.default_max_completion_tokens,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Full,
                SupportTypes.ParallelToolCalling: SupportStrength.Full,
            },
        },
        "gpt-4o-mini": {
            "max_token_size": agent_settings.default_max_completion_tokens,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Full,
                SupportTypes.ParallelToolCalling: SupportStrength.Full,
            },
        },
    },
    Provider.Google.value: {
        "gemini-2.0-flash-exp": {
            "max_token_size": 8_192,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Acceptable,
                SupportTypes.ParallelToolCalling: SupportStrength.NotSupported,
            },
        }
    },
    Provider.Anthropic.value: {
        "claude-3-5-sonnet-20241022": {
            "max_token_size": 8_192,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Acceptable,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Acceptable,
                SupportTypes.ParallelToolCalling: SupportStrength.NotSupported,
            },
        },
        "claude-3-5-haiku-20241022": {
            "max_token_size": 8_192,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.NotSupported,
                SupportTypes.FunctionCalling: SupportStrength.Partially,
                SupportTypes.ParallelToolCalling: SupportStrength.NotSupported,
            },
        },
        "claude-3-opus-20240229": {
            "max_token_size": 4_096,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Partially,
                SupportTypes.ParallelToolCalling: SupportStrength.NotSupported,
            },
        },
        "claude-3-sonnet-20240229": {
            "max_token_size": 4_096,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Partially,
            },
        },
        "claude-3-haiku-20240307": {
            "max_token_size": 4_096,
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.ImageIn: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Partially,
            },
        },
    },
}

AUDIO_SUPPORTED_MODELS = {
    Provider.Google.value: {"gemini-2.0-flash-exp"},
    Provider.OpenAI.value: {
        "gpt-4o-audio-preview",
    },
}

PROVIDERS_ORDER = [
    Provider.OpenAI.value,
    Provider.Google.value,
    Provider.Anthropic.value,
]

__all__ = [
    "PROVIDERS_ORDER",
    "MODELS",
    "AUDIO_SUPPORTED_MODELS",
    "DEFAULT_TEMPERATURE",
    "SupportTypes",
    "SupportStrength",
]
