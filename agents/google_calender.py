from .base import CitexGPT4Agent, ComposioAction, agent_settings

agent = CitexGPT4Agent(
    name="Google Calender Agent",
    tools=agent_settings.composio_tools.get_tools(
        actions=[
            ComposioAction.GOOGLECALENDAR_FIND_FREE_SLOTS,
            ComposioAction.GOOGLECALENDAR_CREATE_EVENT,
            ComposioAction.GMAIL_FETCH_EMAILS,
            ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
            ComposioAction.GMAIL_REPLY_TO_THREAD,
            ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
        ]
    ),
    description=(
        "Analyze google calender, email, fetch emails, and create event on calendar depending on the email content. "
        "You should also draft an email in response to the sender of the previous email. "
        "IMPORTANT: You cannot **send** any email. You can only draft the email. "
        "IMPORTANT: You cannot **reply** to the email on behalf of the sender. You can only draft the email. "
    ),
    delegation_directives=[
        (
            "To analyze google calender, email, fetch emails, and create event on calendar "
            "depending on the email content and write the email as draft, "
            "delegate the task to the `Google Calender Agent`."
        ),
    ],
)


__all__ = ["agent"]
