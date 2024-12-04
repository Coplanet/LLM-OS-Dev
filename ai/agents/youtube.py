from phi.tools.youtube_tools import YouTubeTools

from .base import Agent, AgentConfig

agent_name = "Youtube Agent"


def get_agent(config: AgentConfig = None):
    return Agent(
        name=agent_name,
        agent_config=config,
        tools=[YouTubeTools()],
        description=(
            "You are a YouTube Agent. Fetch the full text or captions of a "
            "YouTube video using the URL, and answer questions."
        ),
        delegation_directives=[
            (
                "To fetch full text or captions of YouTube video using a url, "
                "delegate the task to the `Youtube Agent`."
            ),
        ],
    )


__all__ = ["get_agent", "agent_name"]
