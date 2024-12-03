from textwrap import dedent

from phi.model.base import Model
from phi.utils.log import logger

from .base import Agent, ComposioAction, agent_settings

agent_name = "Google Calender Agent"


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
        tools=agent_settings.composio_tools.get_tools(
            actions=[
                ComposioAction.GOOGLECALENDAR_FIND_FREE_SLOTS,
                ComposioAction.GOOGLECALENDAR_CREATE_EVENT,
                ComposioAction.GOOGLECALENDAR_FIND_EVENT,
                ComposioAction.GOOGLECALENDAR_GET_CALENDAR,
                ComposioAction.GOOGLECALENDAR_LIST_CALENDARS,
                ComposioAction.GOOGLECALENDAR_UPDATE_EVENT,
                ComposioAction.GOOGLECALENDAR_DELETE_EVENT,
                ComposioAction.GMAIL_FETCH_EMAILS,
                ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
                ComposioAction.GMAIL_REPLY_TO_THREAD,
                ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
            ]
        ),
    ).register_or_load(
        default_agent_config={
            "description": dedent(
                """\
                Analyze google calender, email, fetch emails, and create event on calendar depending \
                    on the email content.
                You should also draft an email in response to the sender of the previous email.
                IMPORTANT: You cannot **send** any email. You can only draft the email.
                IMPORTANT: You cannot **reply** to the email on behalf of the sender. You can only draft the email.\
                """
            ),
            "delegation_directives": [
                (
                    "To analyze google calender, email, fetch emails, and create event on calendar "
                    "depending on the email content and write the email as draft, "
                    "delegate the task to the `Google Calender Agent`."
                ),
            ],
        },
        force_model=model,
    )


__all__ = ["get_agent", "agent_name"]
