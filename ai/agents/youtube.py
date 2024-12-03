from phi.tools.youtube_tools import YouTubeTools

from .base import Agent


def get_agent():
    return Agent(
        name="Youtube Agent",
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
    )


__all__ = ["get_agent"]
