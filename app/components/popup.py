import streamlit as st


@st.dialog("Configure Agent")
def show_popup(assistant_name):
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
            st.rerun()

    with col2:
        if st.button("Save"):
            st.session_state.show_popup = False
            st.session_state["generic_leader"] = None
            st.success(f"Settings saved for {assistant_name} Assistant!")
            st.rerun()
