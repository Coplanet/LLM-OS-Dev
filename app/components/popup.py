from collections import OrderedDict

import streamlit as st
from streamlit_pills import pills

from ai.agents.settings import AgentConfig, agent_settings
from app.utils import to_label
from db.session import get_db_context
from db.tables import UserConfig
from helpers.log import logger

DEFAULT_TEMPERATURE = OrderedDict(
    [
        ("Deterministic", {"value": 0.0, "icon": ":material/border_outer:"}),
        ("Balanced", {"value": 0.75, "icon": ":material/balance:"}),
        ("Creative", {"value": 1.5, "icon": ":material/emoji_objects:"}),
    ]
)


def get_temperature(provider: str, model_id: str, selected_temperature: float):
    if provider == "OpenAI" or provider == "Anthropic":
        return {"0.75": 0.5, "1.5": 1}.get(
            str(selected_temperature), selected_temperature
        )
    return selected_temperature


MODELS = {
    "OpenAI": {
        # "gpt-4o-audio-preview": {
        #             #     "max_token_size": agent_settings.default_max_completion_tokens,
        #     "kwargs": {
        #         "modalities": ["text", "audio"],
        #         "audio": {"voice": "alloy", "format": "wav"},
        #     },
        # },
        "gpt-4o": {
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        "gpt-4o-mini": {
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        # "o1-preview": {
        #   "max_token_size": agent_settings.default_max_completion_tokens,
        # },
        # "o1-mini": {
        #   "max_token_size": agent_settings.default_max_completion_tokens,
        # },
    },
    "Google": {
        "gemini-2.0-flash-exp": {
            "max_token_size": 8_192,
        },
    },
    "Groq": {
        "llama-3.3-70b-versatile": {"max_token_size": 32_000},
        "mixtral-8x7b-32768": {
            "max_token_size": 32_000,
        },
    },
    "Anthropic": {
        "claude-3-5-sonnet-20241022": {"max_token_size": 8_192},
        "claude-3-5-haiku-20241022": {"max_token_size": 8_192},
        "claude-3-opus-20240229": {"max_token_size": 4_096},
        "claude-3-sonnet-20240229": {"max_token_size": 4_096},
        "claude-3-haiku-20240307": {"max_token_size": 4_096},
    },
}

AUDIO_SUPPORTED_MODELS = {
    "Google": {"gemini-2.0-flash-exp"},
    "OpenAI": {
        "gpt-4o-audio-preview",
    },
}

PROVIDERS_ORDER = ["OpenAI", "Groq", "Google", "Anthropic"]


@st.dialog("Configure Agent", width="large")
def show_popup(session_id, assistant_name, config: AgentConfig, package):
    label = to_label(assistant_name)

    st.markdown(f"Agent: **{assistant_name}**")

    # Two dropdowns with string options
    provider: str = st.selectbox(
        "Foundation Model Provider", PROVIDERS_ORDER, key=f"{label}_model_type"
    )
    PROVIDER_CONFIG: dict = MODELS[provider]

    if st.session_state[f"{label}_model_id"] not in PROVIDER_CONFIG:
        st.session_state[f"{label}_model_id"] = list(PROVIDER_CONFIG.keys())[0]

    model_id: str = st.selectbox(
        "Model ID", list(PROVIDER_CONFIG.keys()), key=f"{label}_model_id"
    )

    MODEL_CONFIG: dict = PROVIDER_CONFIG[model_id]

    temprature_index = 0
    for i, key in enumerate(DEFAULT_TEMPERATURE.keys()):
        if (
            get_temperature(provider, model_id, DEFAULT_TEMPERATURE[key]["value"])
            == config.temperature
        ):
            temprature_index = i
            break

    st.session_state[f"{label}_temperature"] = config.temperature = temperature = (
        get_temperature(
            provider,
            model_id,
            DEFAULT_TEMPERATURE[
                pills(
                    "Choose the model's creativity:",
                    list(DEFAULT_TEMPERATURE.keys()),
                    index=temprature_index,
                )
            ]["value"],
        )
    )

    max_tokens = MODEL_CONFIG["max_token_size"]

    logger.info(
        "Provider: '%s', Model: '%s', Temperature: '%s', Max Tokens: '%s'",
        provider,
        model_id,
        temperature,
        max_tokens,
    )

    enabled_tools = {}
    available_tools_manifest = {}

    if package.available_tools:
        st.markdown("---")
        st.markdown("### Agent Tools")

        config: UserConfig = None

        with get_db_context() as db:
            config = UserConfig.get_models_config(db, session_id)

        if not config:
            raise ValueError(f"No config found for session: '{session_id}'")

        enabled_tools = config.value_json.get(label, {}).get("tools", {})
        disable_all = enabled_tools.get("disable_all", False)
        enable_all = enabled_tools.get("enable_all", not enabled_tools)

        if not enabled_tools and enable_all:
            for tool in package.available_tools:
                if tool.get("default_status", "enabled").lower() != "enabled":
                    enable_all = False
                    break

            if not enable_all:
                for tool in package.available_tools:
                    key = tool.get("name", None)

                    if "group" in tool:
                        key = f"group:{tool['group']}"

                    if (
                        key not in enabled_tools
                        and tool.get("default_status", "enabled").lower() == "enabled"
                    ):
                        enabled_tools[key] = True

        if not isinstance(package.available_tools, list):
            raise ValueError(
                f"Available tools for '{package.agent_name}' are not a list."
            )

        for tool in package.available_tools:
            if not isinstance(tool, dict):
                raise ValueError(
                    f"Tool '{tool.__name__}' is not a dictionary in package '{package.agent_name}'."
                )

            icon = tool.get("icon", None)
            name = tool.get("name", None)
            key = name

            if "group" in tool:
                key = f"group:{tool['group']}"
                name = tool["group"]

            if name is None:
                raise ValueError(
                    f"Tool '{tool.__name__}' is not a dictionary in package '{package.agent_name}'."
                )

            available_tools_manifest[key] = {
                "name": name,
                "icon": icon,
            }

            if not enable_all:
                if disable_all or key not in enabled_tools:
                    continue

            enabled_tools[key] = (
                enable_all
                or key in enabled_tools
                or tool.get("default_status", "enabled").lower() == "enabled"
            )

        col_index = 0
        super_cols = st.columns([0.33, 0.33, 0.33])

        for key, manifest in available_tools_manifest.items():
            with super_cols[col_index % 3]:
                icon = manifest["icon"]
                name = manifest["name"]
                # Create columns for layout
                col1, col2 = None, None

                if icon:
                    col1, col2 = st.columns([0.1, 0.9])
                else:
                    col1 = st.columns(1)[0]

                with col1:
                    value: str = st.checkbox(
                        name,
                        value=enable_all or (key in enabled_tools),
                        label_visibility="collapsed" if icon else "visible",
                        key=f"checkbox_{key}",
                        help=name,
                    )

                if icon:
                    with col2:
                        st.markdown(
                            f"""
                            <span style="position: relative">
                                <span style="top: 17px; position: absolute; width: 100vw;">
                                    <i class="{icon}" style="margin-right: 10px;"></i> {name}
                                </span>
                            </span>
                            """,
                            unsafe_allow_html=True,
                        )

                if value:
                    enabled_tools[key] = True

                elif key in enabled_tools:
                    del enabled_tools[key]
                    if "enable_all" in enabled_tools:
                        del enabled_tools["enable_all"]

            col_index += 1

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel"):
            st.session_state.show_popup = False
            st.session_state.selected_assistant = None
            st.rerun()

    with col2:
        if st.button("Save"):
            st.session_state.show_popup = False
            st.session_state["generic_leader"] = None

            tools = enabled_tools

            if available_tools_manifest and not tools:
                tools["disable_all"] = True

            if (
                available_tools_manifest
                and tools
                and set(tools.keys()) == set(available_tools_manifest.keys())
            ):
                tools["enable_all"] = True

            new_configs = {
                "model_type": provider,
                "model_id": model_id,
                "model_kwargs": MODEL_CONFIG.get("kwargs", {}),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
            }
            logger.info("New configs: %s", new_configs)
            with get_db_context() as db:
                # get the user
                config = UserConfig.get_models_config(db, session_id)
                if label not in config.value_json:
                    config.value_json[label] = {}
                for key, value in new_configs.items():
                    logger.info("setting %s.%s = %s", label, key, value)
                    config.value_json[label][key] = value

                logger.info("Config to store: %s", config.value)
                # save config in sqlacademy
                config.save(db)
                logger.info(
                    "Stored config: %s",
                    UserConfig.get_models_config(db, session_id).value,
                )

            logger.debug("User configuration stored for session: '%s'", session_id)
            st.success(f"Settings saved for {assistant_name} Assistant!")
            st.session_state.CONFIG_CHANGED = True
            st.rerun()
