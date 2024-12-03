from typing import Callable, Dict

import streamlit as st

from ai.agents.base import Agent
from dashboard.models import UserConfig
from dashboard.models.agent import AgentConfig

from .icons import ICONS


def create_sidebar(session_id, agents: Dict[str, Callable[[], Agent]]) -> None:
    """Create the sidebar with assistant toggles and gear buttons."""
    with st.sidebar:
        for assistant in agents:
            label = assistant.lower().replace(" ", "_")
            feature_name = f"{label}_agent_enabled"

            if feature_name not in st.session_state:
                st.session_state[feature_name] = True

            # Get the icon and agent
            icon = ICONS.get(assistant, "")
            agent_enabled: bool = False

            if assistant in agents:
                agent_enabled = AgentConfig.objects.filter(
                    name=assistant, enabled=True
                ).exists()
            else:
                agent_enabled = True

            # Get pre-value from session state
            pre_value = st.session_state[feature_name] and agent_enabled

            # Create columns for layout
            col1, col2, col3 = st.columns([1, 7, 6])

            with col1:
                new_value = st.checkbox(
                    "label",
                    value=pre_value,
                    key=f"checkbox_{feature_name}",
                    label_visibility="collapsed" if icon else "visible",
                    disabled=not agent_enabled,
                )

            if icon:
                with col2:
                    checkbox_text = assistant.replace("_", " ").title()
                    st.markdown(
                        f"""
                        <span style="position: relative">
                            <span style="top: 27px; position: absolute; width: 100vw;">
                                <i class="{icon}" style="margin-right: 10px;"></i> {checkbox_text}
                            </span>
                        </span>
                        """,
                        unsafe_allow_html=True,
                    )

            with col3:
                st.markdown(
                    f"""
                    <style>
                        .st-key-gear_{label} button {{
                            float: right;
                            background: none; border: none; cursor: pointer; margin-top: -10px; font-size: 18px;
                        }}
                        .st-key-gear_{label} button:after {{ clear: both }}
                    </style>
                """,
                    unsafe_allow_html=True,
                )
                gear_button = st.button("⚙", key=f"gear_{label}")
                if gear_button:
                    st.session_state.selected_assistant = assistant
                    st.session_state.show_popup = True

            if pre_value != new_value:
                st.session_state[feature_name] = new_value
                uc, _ = UserConfig.objects.get_or_create(
                    session_id=session_id, key=feature_name
                )
                uc.value = str(int(new_value))
                uc.save()
                st.rerun()
