import streamlit as st

from ai.agents.settings import AgentConfig, agent_settings
from app.utils import to_label
from db.session import get_db_context
from db.tables import UserConfig
from helpers.log import logger
from helpers.utils import to_title

MODELS = {
    "OpenAI": {
        "gpt-4o": {
            "max_temperature": 2,
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        "gpt-4o-mini": {
            "max_temperature": 2,
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        # "o1-preview": {
        #     "max_temperature": 2,
        #     "max_token_size": agent_settings.default_max_completion_tokens,
        # },
        # "o1-mini": {
        #     "max_temperature": 2,
        #     "max_token_size": agent_settings.default_max_completion_tokens,
        # },
    },
    "Google": {
        "gemini-2.0-flash-exp": {
            "max_temperature": 2,
            "max_token_size": 8_192,
        },
    },
    "Groq": {
        "llama3-groq-70b-8192-tool-use-preview": {
            "max_temperature": 2,
            "max_token_size": 8_192,
        },
        "llama3-groq-8b-8192-tool-use-preview": {
            "max_temperature": 2,
            "max_token_size": 8_192,
        },
        "mixtral-8x7b-32768": {
            "max_temperature": 2,
            "max_token_size": 32_000,
        },
        "llama-3.3-70b-versatile": {"max_temperature": 2, "max_token_size": 32_000},
    },
}

PROVIDERS_ORDER = [
    "OpenAI",
    "Groq",
    "Google",
]


@st.dialog("Configure Agent")
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

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=float(MODEL_CONFIG["max_temperature"]),
        step=0.1,
        key=f"{label}_temperature",
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

        enabled_tools = {tool.__class__: True for tool in package.agent.tools}

        if isinstance(package.available_tools, dict):
            available_tools_manifest = package.available_tools
        else:
            enabled_tools_ = {}
            for tool in package.agent.tools:
                enabled_tools_[tool.name] = True

            for tool in package.available_tools:
                if not isinstance(tool, dict):
                    raise ValueError(
                        f"Tool '{tool.__name__}' is not a dictionary in package '{package.agent_name}'."
                    )

                available_tools_manifest[tool.get("name")] = tool
                if tool["instance"].name in enabled_tools_:
                    enabled_tools[tool.get("name")] = True

        for klass, config in available_tools_manifest.items():
            # Create columns for layout
            col1, col2 = st.columns([1, 11])

            name = config.get(
                "name",
                to_title(klass.__name__) if hasattr(klass, "__name__") else klass,
            )
            icon = config.get("icon", None)
            # Create columns for layout
            col1, col2 = None, None

            if icon:
                col1, col2 = st.columns([0.05, 0.95])
            else:
                col1 = st.columns(1)[0]

            with col1:
                value: str = st.checkbox(
                    name,
                    value=(klass in enabled_tools),
                    label_visibility="collapsed" if icon else "visible",
                    key=f"checkbox_{name}",
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
                enabled_tools[klass] = True

            elif klass in enabled_tools:
                if klass in enabled_tools:
                    del enabled_tools[klass]

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

            tools = {}
            if available_tools_manifest:
                tools = {
                    getattr(k, "__name__", k): True
                    for k in available_tools_manifest.keys()
                    if k in enabled_tools
                }

            new_configs = {
                "model_type": provider,
                "model_id": model_id,
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
            st.rerun()
