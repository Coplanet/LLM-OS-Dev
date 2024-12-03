from phi.agent.python import PythonAgent
from phi.model.base import Model
from phi.utils.log import logger

from .base import Agent
from .settings import extra_settings


class IPythonAgent(Agent, PythonAgent): ...


agent_name = "Python Agent"


def get_agent(model: Model = None):
    if model is not None:
        logger.debug(
            "Agent '%s' uses model: '%s' with temperature: '%s'",
            agent_name,
            model.id,
            str(getattr(model, "temperature", "n/a")),
        )

    return IPythonAgent(
        name=agent_name,
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
        force_model=model,
    )


__all__ = ["get_agent", "agent_name"]
