from textwrap import dedent

from helpers.tool_processor import process_composio_tools
from helpers.utils import to_title

from .base import Agent, AgentConfig, ComposioAction, agent_settings

agent = None
agent_name = "GitHub Agent"

__names = {
    "GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER": "Github: Star Repository"
}
__icons = {
    "GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER": "fa-solid fa-star",
}

available_tools = [
    {
        "instance": instance,
        "name": __names.get(instance.name, to_title(instance.name)),
        "icon": __icons.get(instance.name, to_title(instance.name)),
    }
    for instance in agent_settings.composio_tools.get_tools(
        actions=[
            ComposioAction.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
        ]
    )
]


def get_agent(config: AgentConfig = None):
    tools, _ = process_composio_tools(agent_name, config, available_tools)

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
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

    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
