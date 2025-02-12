import streamlit as st
from streamlit_pills import pills

from ai.agents.base import Agent, Provider
from ai.agents.settings import AgentConfig
from app.models import (
    DEFAULT_TEMPERATURE,
    MODELS,
    PROVIDERS_ORDER,
    SupportStrength,
    SupportTypes,
)
from app.utils import next_run_toast, rerun, to_label
from db.session import get_db_context
from db.tables import UserConfig
from helpers.log import logger


def get_temperature_list(provider: str, model_id: str):
    if provider == Provider.OpenAI.value and model_id in ["o1-mini", "o3-mini"]:
        return ["Creative"]
    return list(DEFAULT_TEMPERATURE.keys())


def get_temperature(provider: str, model_id: str, selected_temperature: float):
    if provider == Provider.OpenAI.value and model_id in ["o1-mini", "o3-mini"]:
        return 1
    if provider == Provider.OpenAI.value or provider == Provider.Anthropic.value:
        return {"0.75": 0.5, "1.5": 1}.get(
            str(selected_temperature), selected_temperature
        )
    return selected_temperature


@st.dialog("Configure Agent", width="large")
def model_config(
    agent: Agent, session_id, assistant_name, config: AgentConfig, package
):
    label = to_label(assistant_name)

    st.markdown(f"Agent: **{assistant_name}**")

    available_models = getattr(package, "available_models", []) or MODELS

    # Two dropdowns with string options
    provider: str = st.selectbox(
        "Foundation Model Provider",
        [i for i in PROVIDERS_ORDER if i in available_models],
        key=f"{label}_model_type",
        disabled=len(available_models) <= 1,
    )

    if (
        provider != agent.model.provider
        and agent.memory.messages
        and not agent.transformer_exists(Provider(provider))
    ):
        st.error(
            "You cannot switch from your current provider **'{}'** to **'{}'**.".format(
                agent.model.provider, provider
            )
        )
        return

    PROVIDER_CONFIG: dict = available_models[provider]

    if st.session_state[f"{label}_model_id"] not in PROVIDER_CONFIG:
        st.session_state[f"{label}_model_id"] = list(PROVIDER_CONFIG.keys())[0]

    model_id: str = st.selectbox(
        "Model ID",
        list(PROVIDER_CONFIG.keys()),
        key=f"{label}_model_id",
        disabled=len(PROVIDER_CONFIG) <= 1,
    )

    MODEL_CONFIG: dict = PROVIDER_CONFIG[model_id]

    SUPPORTS: dict = MODEL_CONFIG.get("supports", {})
    if SUPPORTS:
        st.markdown("### Provider/Model Capabilities:")

        cols = st.columns(3)

        for index, feature in enumerate(SupportTypes):
            support_strength = SupportStrength.NotSupported
            if feature in SUPPORTS:
                support_strength = SUPPORTS[feature]

            # add color to the support strength
            color = support_strength.color()

            with cols[index % len(cols)]:
                st.markdown(
                    f"<strong>{feature.value}</strong>: <span style='color: {color};'>{support_strength.value}</span>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

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
                    get_temperature_list(provider, model_id),
                    index=temprature_index,
                )
            ]["value"],
        )
    )

    max_tokens = MODEL_CONFIG.get("max_token_size")

    logger.info(
        "Provider: '%s', Model: '%s', Temperature: '%s', Max Tokens: '%s'",
        provider,
        model_id,
        temperature,
        max_tokens,
    )

    hidden_tools = {}
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

            if tool.get("hidden", False):
                hidden_tools[key] = True
                enabled_tools[key] = (
                    tool.get("default_status", "enabled").lower() == "enabled"
                )

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
                # if the tool is hidden, skip it
                if key in hidden_tools:
                    continue
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

    if callable(getattr(package, "model_config_modal_ext", None)):
        package.model_config_modal_ext(
            agent, session_id, assistant_name, config, package
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", icon=":material/close:", type="secondary"):
            st.session_state.show_popup = False
            st.session_state.selected_assistant = None
            rerun()

    with col2:
        if st.button("Save", icon=":material/check_circle:", type="primary"):
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

            if callable(getattr(package, "model_config_modal_on_save", None)):
                new_configs_ = package.model_config_modal_on_save(
                    agent, session_id, assistant_name, config, package, new_configs
                )
                if new_configs_:
                    new_configs = new_configs_

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
            next_run_toast(
                f"Settings saved for {assistant_name} Assistant!",
                icon=":material/check_circle:",
            )
            st.session_state.CONFIG_CHANGED = True

            if agent and agent.model.provider != provider:
                agent.transform(Provider(provider))

            rerun()
