from composio_openai import App

from .base import AgentConfig, ComposioAgent

agent = None
agent_name = "Google Calendar Agent"
available_tools = []


def get_agent(config: AgentConfig = None):
    # flake8: noqa: E501

    agent = ComposioAgent(
        app=App.GOOGLECALENDAR,
        user=config.user,
        name=agent_name,
        agent_config=config,
        delegation_directives=[
            (
                f"Delegate any Google Calendar-related operations or API interactions to the `{agent_name}`. "
                "This includes event creation, calendar management, and scheduling operations."
            ),
            (
                "IMPORTANT: All Google Calendar platform interactions, including creating events, updating events, "
                f"calendar management, and availability checking should be handled by the `{agent_name}`."
            ),
            (
                f"IMPORTANT: Use the `{agent_name}` for any task that requires Google Calendar API access "
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
