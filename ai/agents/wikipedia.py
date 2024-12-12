from textwrap import dedent

from phi.tools.wikipedia import WikipediaTools

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig

agent = None
agent_name = "Wikipedia Search Agent"
available_tools = {WikipediaTools: {"name": "Wikipedia"}}


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
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
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
