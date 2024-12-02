from textwrap import dedent

from phi.tools.arxiv_toolkit import ArxivToolkit

from .base import Agent

agent = Agent(
    name="Arxiv Search Agent",
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
)

__all__ = ["agent"]
