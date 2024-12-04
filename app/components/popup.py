import streamlit as st
from phi.utils.log import logger

from app.utils import to_label
from db.session import get_db_context
from db.tables import UserConfig


@st.dialog("Configure Agent")
def show_popup(session_id, assistant_name):
    label = to_label(assistant_name)

    st.markdown(f"Agent: **{assistant_name}**")

    # Two dropdowns with string options
    model_type = st.selectbox(
        "Select Model Type", ["GPT", "Groq", "LLaMA"], key=f"{label}_model_type"
    )
    model_id = st.text_input("Input Model ID", key=f"{label}_model_id")

    # Float input between 0 and 1
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key=f"{label}_temperature",
    )

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1024,
        max_value=1048576,
        step=64,
        key=f"{label}_max_tokens",
    )

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
                "model_type": model_type,
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
