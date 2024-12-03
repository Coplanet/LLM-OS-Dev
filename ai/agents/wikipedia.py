from textwrap import dedent

from phi.model.base import Model
from phi.tools.wikipedia import WikipediaTools
from phi.utils.log import logger

from .base import Agent

agent_name = "Wikipedia Search Agent"


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
        tools=[WikipediaTools()],
    ).register_or_load(
        default_agent_config={
            "description": dedent(
                """
                You are a world-class researcher assigned a very important task.
                Given a topic, search Wikipedia for the most comprehensive and relevant information about that topic.
                Extract and summarize the key points to provide a clear and concise overview.
                This is an important task and your output should be highly relevant and informative \
                    to the original topic.
                """
            ),
            "delegation_directives": [
                (
                    "For searching Wikipedia, "
                    "delegate the task to the `Wikipedia Search Agent`."
                ),
            ],
        },
        force_model=model,
    )


__all__ = ["get_agent", "agent_name"]
