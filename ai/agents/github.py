from .base import ComposioAction, GPT4Agent, agent_settings

agent = GPT4Agent(
    name="GitHub Agent",
    tools=agent_settings.composio_tools.get_tools(
        actions=[
            ComposioAction.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
        ]
    ),
    description=(
        """\
        You are a GitHub assistant capable of starring repositories on behalf of the authenticated user.
        """
    ),
    delegation_directives=[
        "Delegate the task to the `GitHub Agent` for starring repositories on GitHub.",
        (
            "For reading or analyzing a repository, use other enabled tools to crawl the repository and "
            "do not delegate to the `GitHub Agent`."
        ),
    ],
)


__all__ = ["agent"]
