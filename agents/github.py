from .base import CitexGPT4Agent, ComposioAction, agent_settings

agent = CitexGPT4Agent(
    name="GitHub Agent",
    tools=agent_settings.composio_tools.get_tools(
        actions=[
            ComposioAction.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
        ]
    ),
    description=(
        "You are a Github assistant capable of performing tasks on Github, "
        "like make stars, create repositories etc"
    ),
    delegation_directives=[
        (
            "For performing any actions in Github, "
            "delegate the task to the `GitHub Agent`."
        ),
    ],
)


__all__ = ["agent"]
