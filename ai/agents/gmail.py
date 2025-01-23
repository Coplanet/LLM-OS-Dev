from composio_phidata import App

from .base import Agent, AgentConfig

agent = None
agent_name = "Gmail Agent"
available_tools = []
composio_agent = True


def get_agent(config: AgentConfig = None):
    # flake8: noqa: E501

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=Agent.get_tools_as_composio_tools(agent_name, config, App.GMAIL),
        instructions=[
            "Always write HTML formatted emails.",
            "If markdown has been provided to you, convert it to HTML before performing the email operation.",
        ],
        delegation_directives=[
            (
                f"Delegate any Gmail-related operations or API interactions to the `{agent_name}`. "
                "This includes management, and inbox operations."
            ),
            (
                "IMPORTANT: All Gmail platform interactions, including sending emails, managing labels, "
                f"inbox organization, and email searches should be handled by the `{agent_name}`."
            ),
            (
                f"IMPORTANT: Use the `{agent_name}` for any task that requires Gmail API access "
                "or platform interaction, even if it's part of a larger workflow."
            ),
            (
                f"IMPORTANT: You need to provide every detail for the `{agent_name}` to work properly. "
                f"The `{agent_name}` just can work with API access AND NOTHING MORE; Any research or any "
                f"operations prior to calling the `{agent_name}`. and THE END RESULT SHOULD BE PROVIDED TO `{agent_name}`"
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
