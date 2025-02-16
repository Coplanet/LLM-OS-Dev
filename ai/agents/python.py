from phi.agent.python import PythonAgent

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig


class IPythonAgent(Agent, PythonAgent):
    def __init__(self, *args, **kwargs):
        if "model" not in kwargs or kwargs["model"] is None:
            kwargs["model"] = AgentConfig.default_model()

        super().__init__(*args, **kwargs)


agent = None
agent_name = "Python Agent"
available_tools = []


def get_agent(config: AgentConfig = None):
    # flake8: noqa: E501
    tools, _ = process_tools(agent_name, config, available_tools)

    agent = IPythonAgent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        role="Advanced Python Code Writer, Executor, and Analyzer",
        pip_install=True,
        description=(
            "The Python Agent excels at generating, executing, and analyzing complex Python code solutions. "
            "It leverages a robust internal knowledge base and dynamic execution frameworks, enabling the production of "
            "efficient and reliable code while incorporating advanced charting libraries and error-control mechanisms."
        ),
        instructions=[
            "<CIRITICAL INSTRUCTIONS>\n"
            "1. Thoroughly analyze the user prompt to extract complete requirements before generating any code.\n"
            "2. Begin every task by consulting the internal knowledge base for proven patterns and coding best practices.\n"
            "3. Write modular and maintainable code with robust error handling and, where applicable, integrated tests.\n"
            "4. Request clarification immediately if any part of the prompt is ambiguous or incomplete.\n"
            "5. Always include comprehensive logging and debug information alongside complete code outputs.\n"
            "</CIRITICAL INSTRUCTIONS>",
        ],
        delegation_directives=[
            f"<CIRITICAL NOTES FOR DELEGATION OF `{agent_name}`>\n"
            f"a. To write and run Python code, delegate the task to the `{agent_name}`.\n"
            f"b. When delegating tasks to `{agent_name}`, include full execution context, such as conversation history and relevant logs.\n"
            f"c. Always return the complete code and any associated outputs (including error messages) without alterations from `{agent_name}`.\n"
            f"</CIRITICAL NOTES FOR DELEGATION OF `{agent_name}`>",
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
