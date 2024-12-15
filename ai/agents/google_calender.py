from textwrap import dedent

from helpers.tool_processor import process_composio_tools
from helpers.utils import to_title

from .base import Agent, AgentConfig, ComposioAction, agent_settings

agent = None
agent_name = "Google Calender Agent"

__names = {
    "GOOGLECALENDAR_FIND_FREE_SLOTS": "Google Calendar: Find Free Slots",
    "GOOGLECALENDAR_CREATE_EVENT": "Google Calendar: Create Event",
    "GOOGLECALENDAR_FIND_EVENT": "Google Calendar: Find Event",
    "GOOGLECALENDAR_GET_CALENDAR": "Google Calendar: Get Calendar",
    "GOOGLECALENDAR_LIST_CALENDARS": "Google Calendar: List Calendars",
    "GOOGLECALENDAR_UPDATE_EVENT": "Google Calendar: Update Event",
    "GOOGLECALENDAR_DELETE_EVENT": "Google Calendar: Delete Event",
    "GMAIL_FETCH_EMAILS": "Gmail: Fetch Emails",
    "GMAIL_CREATE_EMAIL_DRAFT": "Gmail: Create Email Draft",
    "GMAIL_REPLY_TO_THREAD": "Gmail: Reply To Thread",
}

__icons = {
    "GOOGLECALENDAR_FIND_FREE_SLOTS": "fa-solid fa-calendar-day",
    "GOOGLECALENDAR_CREATE_EVENT": "fa-regular fa-calendar-plus",
    "GOOGLECALENDAR_FIND_EVENT": "fa-solid fa-calendar-week",
    "GOOGLECALENDAR_GET_CALENDAR": "fa-regular fa-calendar-minus",
    "GOOGLECALENDAR_LIST_CALENDARS": "fa-regular fa-calendar-days",
    "GOOGLECALENDAR_UPDATE_EVENT": "fa-regular fa-calendar-check",
    "GOOGLECALENDAR_DELETE_EVENT": "fa-regular fa-calendar-xmark",
    "GMAIL_FETCH_EMAILS": "fa-regular fa-envelope-open",
    "GMAIL_CREATE_EMAIL_DRAFT": "fa-regular fa-envelope",
    "GMAIL_REPLY_TO_THREAD": "fa-solid fa-reply",
}

available_tools = [
    {
        "instance": instance,
        "name": __names.get(instance.name, to_title(instance.name)),
        "icon": __icons.get(instance.name, to_title(instance.name)),
    }
    for instance in agent_settings.composio_tools.get_tools(
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
        ]
    )
]


def get_agent(config: AgentConfig = None):
    tools, _ = process_composio_tools(agent_name, config, available_tools)

    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        description=dedent(
            """\
            Analyze google calender, email, fetch emails, and create event on calendar depending on the email content.
            You should also draft an email in response to the sender of the previous email.
            IMPORTANT: You cannot **send** any email. You can only draft the email.
            IMPORTANT: You cannot **reply** to the email on behalf of the sender. You can only draft the email.\
            """
        ),
        delegation_directives=[
            dedent(
                """\
                To analyze google calender, email, fetch emails, and create event on calendar
                depending on the email content and write the email as draft,
                delegate the task to the `Google Calender Agent`.\
                """
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
