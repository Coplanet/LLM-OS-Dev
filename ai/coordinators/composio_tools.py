from ai.agents.base import ComposioAction

COMPOSIO_ACTIONS = {
    # Gmail
    # "GMAIL": {
    #     "name": "Gmail",
    #     "icon": "fa-solid fa-at",
    #     "actions": [
    #         ComposioAction.GMAIL_ADD_LABEL_TO_EMAIL,
    #         ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
    #         ComposioAction.GMAIL_CREATE_LABEL,
    #         ComposioAction.GMAIL_FETCH_EMAILS,
    #         ComposioAction.GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID,
    #         ComposioAction.GMAIL_FETCH_MESSAGE_BY_THREAD_ID,
    #         ComposioAction.GMAIL_GET_ATTACHMENT,
    #         ComposioAction.GMAIL_GET_PEOPLE,
    #         ComposioAction.GMAIL_GET_PROFILE,
    #         ComposioAction.GMAIL_LIST_LABELS,
    #         ComposioAction.GMAIL_LIST_THREADS,
    #         ComposioAction.GMAIL_MODIFY_THREAD_LABELS,
    #         ComposioAction.GMAIL_REMOVE_LABEL,
    #         ComposioAction.GMAIL_REPLY_TO_THREAD,
    #         ComposioAction.GMAIL_SEND_EMAIL,
    #     ],
    # },
    # Google Calendar
    "GOOGLECALENDAR": {
        "name": "Google Calendar",
        "icon": "fa-solid fa-calendar-day",
        "actions": [
            ComposioAction.GOOGLECALENDAR_CREATE_EVENT,
            ComposioAction.GOOGLECALENDAR_DELETE_EVENT,
            ComposioAction.GOOGLECALENDAR_DUPLICATE_CALENDAR,
            ComposioAction.GOOGLECALENDAR_FIND_EVENT,
            ComposioAction.GOOGLECALENDAR_FIND_FREE_SLOTS,
            ComposioAction.GOOGLECALENDAR_GET_CALENDAR,
            ComposioAction.GOOGLECALENDAR_GET_CURRENT_DATE_TIME,
            ComposioAction.GOOGLECALENDAR_LIST_CALENDARS,
            ComposioAction.GOOGLECALENDAR_PATCH_CALENDAR,
            ComposioAction.GOOGLECALENDAR_QUICK_ADD,
            ComposioAction.GOOGLECALENDAR_REMOVE_ATTENDEE,
            ComposioAction.GOOGLECALENDAR_UPDATE_EVENT,
        ],
    },
    # Github
    "GITHUB": {
        "name": "Github",
        "icon": "fa-brands fa-github",
        "actions": [
            ComposioAction.GITHUB_STAR_A_GIST,
            ComposioAction.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
            ComposioAction.GITHUB_UNSTAR_A_GIST,
            ComposioAction.GITHUB_UNSTAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
            ComposioAction.GITHUB_ISSUES_GET,
            ComposioAction.GITHUB_ISSUES_LIST,
            ComposioAction.GITHUB_GET_A_CODE_SCANNING_ANALYSIS_FOR_A_REPOSITORY,
            ComposioAction.GITHUB_LIST_CODE_SCANNING_ANALYSES_FOR_A_REPOSITORY,
            ComposioAction.GITHUB_LIST_COMMITS,
            ComposioAction.GITHUB_LIST_COMMITS_ON_A_PULL_REQUEST,
            ComposioAction.GITHUB_LIST_COMMIT_COMMENTS,
            ComposioAction.GITHUB_LIST_COMMIT_COMMENTS_FOR_A_REPOSITORY,
        ],
    },
}


__all__ = ["COMPOSIO_ACTIONS"]
