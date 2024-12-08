import streamlit as st
from phi.utils.log import logger

from ai.agents.settings import agent_settings
from app.utils import to_label
from db.session import get_db_context
from db.tables import UserConfig

MODELS = {
    "OpenAI": {
        "gpt-4o": {
            "max_temperature": 2,
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        "gpt-4o-Mini": {
            "max_temperature": 2,
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
        # "o1-preview": {
        #     "max_temperature": 2,
        #     "max_token_size": 128_000,
        # },
        # "o1-mini": {
        #     "max_temperature": 2,
        #     "max_token_size": 128_000,
        # },
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
        "gemma-7b-it": {
            "max_temperature": 2,
            "max_token_size": 8_192,
        },
        "llama-3.3-70b-versatile": {"max_temperature": 2, "max_token_size": 128_000},
    },
}

PROVIDERS_ORDER = ["OpenAI", "Groq"]


@st.dialog("Configure Agent")
def show_popup(session_id, assistant_name):
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
            new_configs = {
                "model_type": provider,
                "model_id": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
