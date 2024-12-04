from textwrap import dedent

from .base import Agent, AgentConfig, ComposioAction, agent_settings

agent_name = "GitHub Agent"


def get_agent(config: AgentConfig = None):
    return Agent(
        name=agent_name,
        agent_config=config,
        tools=agent_settings.composio_tools.get_tools(
            actions=[
                ComposioAction.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
            ]
        ),
        description=dedent(
            """\
            You are a GitHub assistant capable of starring repositories on behalf of the authenticated user.
            """
        ),
        delegation_directives=[
            "Delegate the task to the `GitHub Agent` for starring repositories on GitHub.",
            (
                "For reading or analyzing a repository, use other enabled tools to crawl the repository and "
                "do not delegate to the `GitHub Agent`."
            ),
        ],
    )


__all__ = ["get_agent", "agent_name"]
