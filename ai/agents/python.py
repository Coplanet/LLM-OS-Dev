from phi.agent.python import PythonAgent

from .base import Agent
from .settings import extra_settings


class IPythonAgent(Agent, PythonAgent): ...


def get_agent():
    return IPythonAgent(
        name="Python Agent",
        role="Advanced Python Code Writer and Executor",
        pip_install=True,
        base_dir=extra_settings.scratch_dir,
    ).register_or_load(
        default_agent_config={
            "description": (
                "The Python Agent is a specialized tool designed to not only write "
                "advanced professional Python code but also execute it seamlessly. "
                "It supports various charting libraries, including Pandas, matplotlib, numpy, seaborn, "
                "Streamlit, to facilitate data visualization and analysis."
            ),
            "charting_libraries": ["streamlit", "plotly", "matplotlib", "seaborn"],
            "delegation_directives": [
                "To write and run Python code, delegate the task to the `Python Agent`."
            ],
        },
    )


__all__ = ["get_agent"]
