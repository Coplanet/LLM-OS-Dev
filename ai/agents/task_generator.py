from textwrap import dedent

from agno.tools.exa import ExaTools

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig

agent = None
agent_name = "Task Writer Agent"
available_tools = [
    {
        "instance": ExaTools(num_results=5, text_length_limit=1000),
        "name": "Search (Exa)",
        "icon": "fa-solid fa-magnifying-glass",
    }
]


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        role="Jira Task writer agent",
        description=("You are a specialized in writing tasks for Jira."),
        instructions=[
            "You will receive a user prompt which is a content of email or communication or any other form of communication or desired output from the user.",
            "You will conevrt that content into a Jira task's content.",
            "**IMPORTANT:** Be caureful with if that reqires R&D or not, if so write the task in R&D mindset."
            "**IMPORTANT:** If the task requires documentation it needs to be explicitly mentioned in the task's content. if not, don't mention documentation in the task's content.",
        ],
        expected_output=dedent(
            """
            Title: The title of the task.

            {Description of the task}

            Action Steps:
            Numbered list of actions to take in order to finish the task.

            Deliverables:
            Numbered list of items that can be delivered at the end of this task

            Additional Notes: (This section is optional add it if there is any)
            Numbered list of additional notes
            """
        ).strip(),
        delegation_directives=[
            f"Delegate to `{agent_name}` for writing the task for Jira or any other task management tool.",
            (
                f"DO NOT MODIFY THE OUTPUT OF THE TASK YOU RECEIVE FROM `{agent_name}`, JUST RECEIVE IT AND "
                f"RETURN TO THE USER, The `{agent_name}` IS RESPONSIBLE FOR WRITING THE TASK AND KNOWS THE BEST."
            ),
            f"If user requires any task breakdown for given input, delegate it to `{agent_name}`.",
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
