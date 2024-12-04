from textwrap import dedent

from phi.tools.wikipedia import WikipediaTools

from .base import Agent, AgentConfig

agent_name = "Wikipedia Search Agent"


def get_agent(config: AgentConfig = None):
    return Agent(
        name=agent_name,
        agent_config=config,
        tools=[WikipediaTools()],
        description=dedent(
            """\
            You are a world-class researcher assigned a very important task.
            Given a topic, search Wikipedia for the most comprehensive and relevant information about that topic.
            Extract and summarize the key points to provide a clear and concise overview.
            This is an important task and your output should be highly relevant and informative to the original topic.\
            """
        ),
        delegation_directives=[
            (
                "For searching Wikipedia, "
                "delegate the task to the `Wikipedia Search Agent`."
            ),
        ],
    )


__all__ = ["get_agent", "agent_name"]
