from typing import Callable, Dict

import streamlit as st

from ai.agents.base import Agent
from app.utils import rerun
from db.session import get_db_context
from db.tables import UserConfig

from .icons import ICONS


def create_sidebar(session_id, agents: Dict[str, Callable[[], Agent]]) -> None:
    """Create the sidebar with assistant toggles and gear buttons."""
    config: UserConfig = None
    with get_db_context() as db:
        config = UserConfig.get_models_config(db, session_id)

    with st.sidebar:
        for agent, sidebar_config in agents.items():
            label = sidebar_config.get("label", agent)
            icon = sidebar_config.get("icon", ICONS.get(agent, ""))
            selectable = sidebar_config.get("selectable", True)
            feature_name = f"{label}_agent_enabled"

            if feature_name not in st.session_state:
                st.session_state[feature_name] = True

            agent_enabled: bool = True

            if (
                config
                and isinstance(config.value_json, dict)
                and isinstance(config.value_json.get(label), dict)
            ):
                agent_enabled = config.value_json[label].get("enabled", True)

            # Get pre-value from session state
            pre_value = st.session_state[feature_name] and agent_enabled

            # Create columns for layout
            col1, col2, col3 = st.columns([0.1, 0.7, 0.15])

            with col1:
                new_value = True
                if selectable:
                    new_value = st.checkbox(
                        "label",
                        value=pre_value,
                        key=f"checkbox_{feature_name}",
                        label_visibility="collapsed" if icon else "visible",
                    )

            if icon:
                with col2:
                    checkbox_text = agent.replace("_", " ").title()
                    st.markdown(
                        f"""
                        <span style="position: relative">
                            <span style="top: 17px; position: absolute; width: 100vw;">
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
                            background: none; border: none; cursor: pointer; margin-top: -20px; font-size: 18px;
                        }}
                        .st-key-gear_{label} button:after {{ clear: both }}
                    </style>
                """,
                    unsafe_allow_html=True,
                )
                gear_button = st.button("⚙", key=f"gear_{label}")
                if gear_button:
                    st.session_state.selected_assistant = agent
                    st.session_state.show_popup = True

            if pre_value != new_value:
                st.session_state[feature_name] = new_value
                if config:
                    if label not in config.value_json:
                        config.value_json[label] = {}

                    config.value_json[label]["enabled"] = bool(new_value)

                    with get_db_context() as db:
                        config.save(db)
                rerun()
