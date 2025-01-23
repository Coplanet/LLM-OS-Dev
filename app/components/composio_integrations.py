import hashlib

import streamlit as st
from composio import App, ComposioToolSet

from app.auth import User
from app.utils import rerun, run_js
from db.tables.user_config import UserIntegration
from workspace.settings import extra_settings

AVAILABLE_APPS = {
    App.GMAIL: {"name": "Gmail", "icon": "fa-solid fa-envelope"},
    App.GITHUB: {"name": "Github", "icon": "fa-brands fa-github"},
    App.TWITTER: {"name": "Twitter", "icon": "fa-brands fa-twitter"},
    App.GOOGLECALENDAR: {"name": "Google Calendar", "icon": "fa-solid fa-calendar"},
}

NAME_TO_APP = {details["name"]: app for app, details in AVAILABLE_APPS.items()}


@st.dialog("Account integrations", width="large")
def composio_integrations(user: User):
    st.subheader("Integrate your account with Composio")
    st.markdown("Integrate your account with Composio to enable AI-powered actions.")
    st.markdown("---")
    integrations = UserIntegration.get_integrations(user.user_id)

    connected_apps = {i.app for i in integrations}

    to_be_connected_apps = set(
        [
            details["name"]
            for app, details in AVAILABLE_APPS.items()
            if app not in connected_apps
        ]
    )

    cols = st.columns(2)
    with cols[1]:
        if integrations.count() > 0:
            st.markdown("### Connected apps")
            st.caption(
                "Select the app to disconnect your account from LLM-OS and Composio"
            )
            for integration in integrations:
                connected_apps.add(integration.app)
                details = AVAILABLE_APPS[integration.app]
                subcols = st.columns([0.1, 0.9])
                delete_integration = False
                with subcols[0]:
                    delete_integration = st.checkbox(
                        integration.app,
                        value=False,
                        label_visibility="collapsed",
                        help="Select to disconnect",
                    )
                with subcols[1]:
                    st.markdown(
                        f"<span class='{details['icon']}' style='margin-right: 5px;'></span> {details['name']}",
                        unsafe_allow_html=True,
                    )

                if delete_integration:
                    with st.spinner("Disconnecting..."):
                        import requests

                        url = "https://backend.composio.dev/api/v1/connectedAccounts/{}".format(
                            integration.connection_id
                        )
                        headers = {"x-api-key": extra_settings.composio_api_key}
                        response = requests.request("DELETE", url, headers=headers)
                        if response.status_code == 200:
                            integration.delete()
                            to_be_connected_apps.add(integration.app)
                            st.success("Disconnected from " + integration.app + "!")
                        else:
                            st.error(
                                "Failed to disconnect from " + integration.app + "!"
                            )

        else:
            st.warning("No apps connected yet; select an app to connect!")

    with cols[0]:
        if to_be_connected_apps:
            st.subheader("Connect an app")
            SELECTED_APP = st.selectbox(
                "Select an app",
                options=to_be_connected_apps,
                index=None,
                placeholder="Choose an app",
            )

            if SELECTED_APP:
                APP = NAME_TO_APP[SELECTED_APP]

                response = None

                with st.spinner("Getting expected parameters..."):
                    toolset = ComposioToolSet()
                    response = toolset.get_expected_params_for_user(app=APP)

                expected_params = response["expected_params"]
                collected_params = {}

                if len(expected_params) > 0:
                    for param in expected_params:
                        user_input = st.text_input(
                            f"Enter the value for '{param.displayName}'"
                        )
                        st.caption(param.description)
                        if user_input:
                            collected_params[param.name] = user_input

                with st.spinner("Initiating connection..."):
                    redirect_url = extra_settings.get_redirect_url(
                        {
                            "app": APP,
                            "source": "composio",
                            "hash": hashlib.sha256(
                                "{}:{}:{}:{}:composio".format(
                                    extra_settings.secret_key,
                                    user.username,
                                    user.session_id,
                                    APP,
                                ).encode()
                            ).hexdigest(),
                        }
                        | user.to_auth_param(False)
                    )
                    # Initiate the connection
                    connection_request = toolset.initiate_connection(
                        connected_account_params=collected_params,
                        entity_id=user.user_id,
                        app=APP,
                        redirect_url=redirect_url,
                    )

                if connection_request.connectionStatus == "INITIATED":
                    # flake8: noqa: E501

                    st.link_button(
                        "Connect to " + SELECTED_APP,
                        url=connection_request.redirectUrl,
                        icon=":material/check:",
                        type="primary",
                    )

                    run_js(
                        f"""
                        link_interval_value_{APP} = setInterval(function() {{
                            // Try to find the anchor element with the specified attributes
                            var anchor = window.parent.document.querySelector('a[kind=primary][target="_blank"][href="{connection_request.redirectUrl}"]');

                            // If the anchor element is found, change its target attribute
                            if (anchor) {{
                                anchor.setAttribute("target", "_self");
                                clearInterval(link_interval_value_{APP});
                            }}
                        }}, 100);
                        """,
                        False,
                    )

                elif connection_request.connectionStatus == "ACTIVE":
                    st.success("Connected to " + SELECTED_APP + "!")
                    # active connection means the user has completed the authentication process.
                    # the API Key entered might still be invalid, you can test by calling the tool.
                else:
                    st.error("Connection process failed, please try again.")
        else:
            st.success(
                ":material/check: **Congratulations!**\n\nYou have connected all the apps!"
            )

    st.markdown("---")
    if st.button("Close", icon=":material/close:", type="primary"):
        rerun()
