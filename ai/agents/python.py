from phi.agent.python import PythonAgent

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig
from .settings import extra_settings


class IPythonAgent(Agent, PythonAgent):
    def __init__(self, *args, **kwargs):
        if "model" not in kwargs or kwargs["model"] is None:
            kwargs["model"] = self.default_model()

        super().__init__(*args, **kwargs)


agent = None
agent_name = "Python Agent"
available_tools = []


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        role="Advanced Python Code Writer and Executor",
        pip_install=True,
        description=(
            "The Python Agent is a specialized tool designed to not only write "
            "advanced professional Python code but also execute it seamlessly. "
            "It supports various charting libraries, including Pandas, matplotlib, numpy, seaborn, "
            "Streamlit, to facilitate data visualization and analysis."
        ),
        charting_libraries=["streamlit", "plotly", "matplotlib", "seaborn"],
        base_dir=extra_settings.scratch_dir,
        delegation_directives=[
            "To write and run Python code, delegate the task to the `Python Agent`."
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
