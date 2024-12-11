from textwrap import dedent

from phi.tools.arxiv_toolkit import ArxivToolkit

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig

agent = None
agent_name = "Arxiv Search Agent"
available_tools = {ArxivToolkit: {"name": "Arxiv"}}


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        description=dedent(
            """\
            You are a world-class researcher assigned which is resposible for searching and retrieving articles from ArXiv and sciencetific schollars.
            """
        ),
        instructions=dedent(
            """\
            You are a world-class researcher assigned a very important task.

            - Given a topic, search ArXiv for the top 10 articles about that topic and return the 3 most relevant articles use `search_arxiv_and_return_articles` in your tools.
            - You can also read a list of arxiv papers and return the content of the papers use `read_arxiv_papers` in your tools.

            This is an important task and your output should be highly relevant to the original topic.\
            """
        ),
        delegation_directives=[
            (
                "For any task involving searching, retrieving, or processing articles from ArXiv or scientific scholars, "
                f"including combining insights from multiple ArXiv documents, transfer the task to the `{agent_name}`. "
                "This includes tasks that require innovative synthesis of ideas "
                "from ArXiv documents, such as creating new patents or combining concepts from different papers."
            ),
            (
                "Ensure that any communication or interaction with ArXiv documents, whether it involves reading, summarizing, "
                f"or analyzing, is delegated to the `{agent_name}`. This agent is equipped to handle all aspects of ArXiv "
                "document processing and should be utilized for any related tasks."
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
