from typing import Literal, Optional

import streamlit as st

from app.utils import rerun
from helpers.utils import to_title
from workspace.settings import extra_settings


@st.dialog("Access Computer User Interface", width="large")
def computer_use(platform: Optional[Literal["gemini", "anthropic"]] = None):
    if not platform:
        if not (
            extra_settings.anthropic_computer_url_link
            and extra_settings.gemini_computer_url_link
        ):
            if extra_settings.anthropic_computer_url_link:
                platform = "anthropic"
            elif extra_settings.gemini_computer_url_link:
                platform = "gemini"
            else:
                st.error("No computer user interface is available")
                return

        if not platform:
            st.subheader("Please select the platform you want to access")
            platform = st.selectbox(
                "Select Platform",
                ["Gemini", "Anthropic"],
                index=None,
            )
            if platform:
                platform = platform.lower()

    if platform:
        PLATFORM_LINK = {
            "gemini": extra_settings.gemini_computer_url_link,
            "anthropic": extra_settings.anthropic_computer_url_link,
        }.get(platform, extra_settings.gemini_computer_url_link)

        PLATFORM_TITLE = to_title(platform)

        st.subheader(
            "Confirm Access to the {}'s Computer User Interface".format(PLATFORM_TITLE)
        )
        st.markdown(
            "This action will open the {}'s computer use's interface, allowing the LLM to perform tasks "
            "and actions in a safe and secure isolated environment. Please ensure you have saved all "
            "necessary work before proceeding.".format(PLATFORM_TITLE)
        )
        cols = st.columns(2)

        with cols[1]:
            st.link_button(
                "Yes",
                PLATFORM_LINK,
                icon=":material/check:",
                type="primary",
            )

        with cols[0]:
            if st.button("No", icon=":material/close:"):
                rerun()
