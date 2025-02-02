from agno.tools.python import PythonTools

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig
from .settings import extra_settings

agent = None
agent_name = "Python Agent"
available_tools = [
    {
        "instance": PythonTools(
            base_dir=extra_settings.scratch_dir,
            save_and_run=True,
            pip_install=True,
            run_code=True,
            list_files=True,
            run_files=True,
            read_files=True,
        ),
        "name": "Python",
        "icon": "fa-brands fa-python",
    }
]


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        role="Advanced Python Code Writer and Executor",
        description=(
            "The Python Agent is a specialized tool designed to not only write "
            "advanced professional Python code but also execute it seamlessly. "
            "It supports various charting libraries, including Pandas, matplotlib, numpy, seaborn, "
            "Streamlit, to facilitate data visualization and analysis."
        ),
        delegation_directives=[
            "To write and run Python code, delegate the task to the `Python Agent`."
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
