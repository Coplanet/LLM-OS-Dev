import streamlit as st

from app.utils import rerun
from workspace.settings import extra_settings


@st.dialog("Access Computer User Interface", width="large")
def computer_use():
    st.subheader("Confirm Access to the Computer User Interface")
    st.markdown(
        "This action will open the computer use's interface, allowing the LLM to perform tasks "
        "and actions in a safe and secure isolated environment. Please ensure you have saved all "
        "necessary work before proceeding."
    )
    cols = st.columns(2)
    with cols[1]:
        st.link_button(
            "Yes",
            extra_settings.computer_url_link,
            icon=":material/check:",
            type="primary",
        )

    with cols[0]:
        if st.button("No", icon=":material/close:"):
            rerun()
