from textwrap import dedent

from phi.model.base import Model
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.utils.log import logger

from .base import Agent

agent_name = "Arxiv Search Agent"


def get_agent(model: Model = None):
    if model is not None:
        logger.debug(
            "Agent '%s' uses model: '%s' with temperature: '%s'",
            agent_name,
            model.id,
            str(getattr(model, "temperature", "n/a")),
        )

    return Agent(
        name=agent_name,
        tools=[ArxivToolkit()],
    ).register_or_load(
        default_agent_config={
            "description": dedent(
                """\
                You are a world-class researcher assigned a very important task.
                Given a topic, search ArXiv for the top 10 articles about that topic and return the 3 most \
                    relevant articles.
                This is an important task and your output should be highly relevant to the original topic.\
                """
            ),
            "delegation_directives": [
                (
                    "For searching ArXiv and sciencetific schollars, "
                    "delegate the task to the `Arxiv Search Agent`."
                ),
            ],
        },
        force_model=model,
    )


__all__ = ["get_agent", "agent_name"]
