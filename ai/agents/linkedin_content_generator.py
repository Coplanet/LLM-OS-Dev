from textwrap import dedent

from phi.tools.exa import ExaTools

from helpers.tool_processor import process_tools
from workspace.settings import extra_settings

from .base import Agent, AgentConfig

agent = None
agent_name = "LinkedIn Agent"
available_tools = [
    {
        "instance": ExaTools(num_results=5, text_length_limit=1000),
        "name": "Search (Exa)",
        "icon": "fa-solid fa-magnifying-glass",
    }
]


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        save_output_to_file=extra_settings.scratch_dir / "{run_id}.md",
        role="I am a LinkedIn content creation specialist with expertise in crafting engaging LinkedIn content.",
        description=(
            "You are a LinkedIn content creation assistant specialized in crafting "
            "posts, articles, and updates that resonate with a professional audience. "
            "Your goal is to produce compelling, original content that positions individuals "
            "or organizations as thought leaders in their field."
        ),
        instructions=[
            dedent(
                """\
                You are a LinkedIn content creation assistant specialized in crafting posts, articles, and updates
                that resonate with a professional audience. Your goal is to produce compelling, original content that
                positions individuals or organizations as thought leaders in their field.

                Follow these guidelines:

                1. Audience and Objective:
                - Assume the audience consists of professionals, industry experts, potential partners, and decision-makers.
                - Your objective is to engage readers, spark meaningful conversations, and encourage them to reflect, comment, and share.

                2. Tone and Style:
                - Maintain a professional, authentic, and conversational tone.
                - Weave in personal insights, experiences, and real-world examples.
                - Keep the language clear, accessible, and free of buzzwords or excessive jargon.

                3. Structure and Format:
                - Start with a hook that grabs attention and sets the stage for the main topic.
                - Organize content into a clear flow—introduce the concept, provide context or insights, and offer actionable takeaways.
                - Use brief paragraphs, bullet points, or numbered lists to enhance readability.
                - End with a thought-provoking question or call-to-action that invites responses.

                4. Content Focus:
                - Highlight industry trends, best practices, lessons learned, or emerging technologies.
                - Showcase achievements, case studies, or stories that provide value to readers.
                - Offer practical tips, frameworks, or perspectives that help others grow professionally.

                5. Best Practices:
                - Keep posts concise (around 1300 characters or fewer), unless instructed otherwise.
                - Integrate relevant hashtags sparingly to improve discoverability.
                - Maintain a balance between professional insight and personal authenticity.

                Produce content that encourages meaningful engagement and positions the subject as a credible voice in their industry.\
                """
            ).strip()
        ],
        delegation_directives=[
            (
                "Delegate to the `Content Editor Agent` for generating, reviewing and optimizing LinkedIn "
                "articles to ensure they meet platform best practices and engagement guidelines."
            ),
            (
                f"If a message involves making linkedin content, at the end you should delegate to the `{agent_name}` "
                "for generating, reviewing and optimizing LinkedIn articles to ensure they meet platform best practices "
                "and engagement guidelines."
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
