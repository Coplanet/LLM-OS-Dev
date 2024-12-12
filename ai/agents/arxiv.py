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

            - When provided with specific ArXiv URLs, prioritize retrieving and processing those documents directly using `read_arxiv_papers` in your tools.
            - Given a general topic without specific URLs, search ArXiv for the top 10 articles about that topic and return the 3 most relevant articles using `search_arxiv_and_return_articles` in your tools.
            - Ensure that your output is highly relevant to the original topic or task request.

            This is an important task, and your focus should be on delivering precise and relevant information based on the user's request.\
            """
        ),
        delegation_directives=[
            (
                "For any task involving searching, retrieving, or processing articles from ArXiv or scientific scholars, "
                f"including combining insights from multiple ArXiv documents, transfer the task to the `{agent_name}`. "
                "This includes tasks that require innovative synthesis of ideas "
                "from ArXiv documents, such as combining concepts from different papers."
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
