from phi.tools.youtube_tools import YouTubeTools

from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig

agent = None
agent_name = "Youtube Agent"
available_tools = {YouTubeTools: {"name": "YouTube"}}


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        description=(
            "You are a YouTube agent that has the special skill of understanding "
            "YouTube videos and answering questions about them."
        ),
        instructions=[
            "Using a video URL, get the video data using the `get_youtube_video_data` tool and captions using the `get_youtube_video_data` tool.",
            "Using the data and captions, answer the user's question in an engaging and thoughtful manner. Focus on the most important details.",
            "If you cannot find the answer in the video, say so and ask the user to provide more details.",
            "Keep your answers concise and engaging.",
            "If the user just provides a URL, summarize the video and answer questions about it.",
        ],
        delegation_directives=[
            (
                f"To analyze YouTube videos, delegate to the `{agent_name}`. `{agent_name}` can:\n"
                "- Extract and analyze video captions and metadata from YouTube URLs\n"
                "- Provide concise video summaries and key points\n"
                "- Answer specific questions about video content\n"
                "- Handle both direct video analysis and Q&A about video content\n"
            ),
            f"Note: The `{agent_name}` works best with videos that have captions available.",
            f"Do NOT delegate non-YouTube video analysis tasks to `{agent_name}` agent.",
        ],
        add_history_to_messages=True,
        num_history_responses=5,
        add_name_to_instructions=True,
        add_datetime_to_instructions=True,
        markdown=True,
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
