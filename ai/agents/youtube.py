from phi.model.base import Model
from phi.tools.youtube_tools import YouTubeTools
from phi.utils.log import logger

from .base import Agent

agent_name = "Youtube Agent"


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
        tools=[YouTubeTools()],
    ).register_or_load(
        default_agent_config={
            "description": (
                "You are a YouTube Agent. Fetch the full text or captions of a "
                "YouTube video using the URL, and answer questions."
            ),
            "delegation_directives": [
                (
                    "To fetch full text or captions of YouTube video using a url, "
                    "delegate the task to the `Youtube Agent`."
                ),
            ],
        },
        force_model=model,
    )


__all__ = ["get_agent", "agent_name"]
