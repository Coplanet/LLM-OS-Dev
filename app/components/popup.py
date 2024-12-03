import streamlit as st
from phi.utils.log import logger

from dashboard.models import UserConfig


@st.dialog("Configure Agent")
def show_popup(session_id, assistant_name):
    label = assistant_name.lower().replace(" ", "_")

    st.markdown(f"Agent: **{assistant_name}**")

    # Two dropdowns with string options
    st.selectbox(
        "Select Model Type", ["GPT", "Groq", "LLaMA"], key=f"{label}_model_type"
    )
    st.text_input("Input Model ID", key=f"{label}_model_id")

    # Float input between 0 and 1
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key=f"{label}_temperature",
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
            tobecreated = {
                f"{label}_model_type",
                f"{label}_model_id",
                f"{label}_temperature",
            }
            uc = UserConfig.objects.filter(
                session_id=session_id, key__in=list(tobecreated)
            )
            for config in uc:
                config.value = st.session_state[config.key]
                del tobecreated[config.key]

            UserConfig.objects.bulk_update(uc, ["value"])

            bulk_create = []

            for key in tobecreated:
                bulk_create.append(
                    UserConfig(
                        session_id=session_id,
                        key=key,
                        value=str(st.session_state[key]),
                    )
                )

            UserConfig.objects.bulk_create(bulk_create)
            logger.debug("User configuration stored for session: '%s'", session_id)
            st.success(f"Settings saved for {assistant_name} Assistant!")
            st.rerun()
