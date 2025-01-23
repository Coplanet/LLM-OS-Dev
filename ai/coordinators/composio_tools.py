from composio import App

from ai.agents.base import ComposioAction

COMPOSIO_ACTIONS = {
    # Gmail
    App.GMAIL: {
        "name": "Gmail",
        "icon": "fa-solid fa-at",
        "actions": [
            ComposioAction.GMAIL_ADD_LABEL_TO_EMAIL,
            ComposioAction.GMAIL_CREATE_EMAIL_DRAFT,
            ComposioAction.GMAIL_CREATE_LABEL,
            ComposioAction.GMAIL_FETCH_EMAILS,
            ComposioAction.GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID,
            ComposioAction.GMAIL_FETCH_MESSAGE_BY_THREAD_ID,
            ComposioAction.GMAIL_GET_ATTACHMENT,
            ComposioAction.GMAIL_GET_PEOPLE,
            ComposioAction.GMAIL_GET_PROFILE,
            ComposioAction.GMAIL_LIST_LABELS,
            ComposioAction.GMAIL_LIST_THREADS,
            ComposioAction.GMAIL_MODIFY_THREAD_LABELS,
            ComposioAction.GMAIL_REMOVE_LABEL,
            ComposioAction.GMAIL_REPLY_TO_THREAD,
            ComposioAction.GMAIL_SEND_EMAIL,
        ],
    },
    # Google Calendar
    App.GOOGLECALENDAR: {
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
    App.GITHUB: {
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
    App.TWITTER: {
        "name": "Twitter",
        "icon": "fa-brands fa-twitter",
        "actions": [
            ComposioAction.TWITTER_ADD_A_LIST_MEMBER,
            ComposioAction.TWITTER_ADD_POST_TO_BOOKMARKS,
            ComposioAction.TWITTER_BOOKMARKS_BY_USER,
            ComposioAction.TWITTER_CAUSES_THE_USER_IN_THE_PATH_TO_LIKE_THE_SPECIFIED_POST,
            ComposioAction.TWITTER_CAUSES_THE_USER_IN_THE_PATH_TO_REPOST_THE_SPECIFIED_POST,
            ComposioAction.TWITTER_CAUSES_THE_USER_IN_THE_PATH_TO_UNLIKE_THE_SPECIFIED_POST,
            ComposioAction.TWITTER_CAUSES_THE_USER_IN_THE_PATH_TO_UNRETWEET_THE_SPECIFIED_POST,
            ComposioAction.TWITTER_CREATE_A_NEW_DM_CONVERSATION,
            ComposioAction.TWITTER_CREATE_LIST,
            ComposioAction.TWITTER_CREATION_OF_A_POST,
            ComposioAction.TWITTER_DELETE_DM,
            ComposioAction.TWITTER_DELETE_LIST,
            ComposioAction.TWITTER_FETCH_LIST_MEMBERS_BY_ID,
            ComposioAction.TWITTER_FETCH_SPACE_TICKET_BUYERS_LIST,
            ComposioAction.TWITTER_FOLLOWERS_BY_USER_ID,
            ComposioAction.TWITTER_FOLLOWING_BY_USER_ID,
            ComposioAction.TWITTER_FOLLOW_A_LIST,
            ComposioAction.TWITTER_FOLLOW_USER,
            ComposioAction.TWITTER_FULL_ARCHIVE_SEARCH,
            ComposioAction.TWITTER_FULL_ARCHIVE_SEARCH_COUNTS,
            ComposioAction.TWITTER_GET_A_USER_S_LIST_MEMBERSHIPS,
            ComposioAction.TWITTER_GET_A_USER_S_OWNED_LISTS,
            ComposioAction.TWITTER_GET_A_USER_S_PINNED_LISTS,
            ComposioAction.TWITTER_GET_DM_EVENTS_BY_ID,
            ComposioAction.TWITTER_GET_DM_EVENTS_FOR_A_DM_CONVERSATION,
            ComposioAction.TWITTER_GET_RECENT_DM_EVENTS,
            ComposioAction.TWITTER_GET_USER_S_FOLLOWED_LISTS,
            ComposioAction.TWITTER_HIDE_REPLIES,
            ComposioAction.TWITTER_LIST_LOOKUP_BY_LIST_ID,
            ComposioAction.TWITTER_LIST_POSTS_TIMELINE_BY_LIST_ID,
            ComposioAction.TWITTER_MUTE_USER_BY_USER_ID,
            ComposioAction.TWITTER_PIN_A_LIST,
            ComposioAction.TWITTER_POSTS_LABEL_STREAM,
            ComposioAction.TWITTER_POST_DELETE_BY_POST_ID,
            ComposioAction.TWITTER_POST_LOOKUP_BY_POST_ID,
            ComposioAction.TWITTER_POST_LOOKUP_BY_POST_IDS,
            ComposioAction.TWITTER_POST_USAGE,
            ComposioAction.TWITTER_RECENT_SEARCH,
            ComposioAction.TWITTER_RECENT_SEARCH_COUNTS,
            ComposioAction.TWITTER_REMOVE_A_BOOKMARKED_POST,
            ComposioAction.TWITTER_REMOVE_A_LIST_MEMBER,
            ComposioAction.TWITTER_RETRIEVE_DM_CONVERSATION_EVENTS,
            ComposioAction.TWITTER_RETRIEVE_POSTS_FROM_A_SPACE,
            ComposioAction.TWITTER_RETRIEVE_POSTS_THAT_QUOTE_A_POST,
            ComposioAction.TWITTER_RETRIEVE_POSTS_THAT_REPOST_A_POST,
            ComposioAction.TWITTER_RETURNS_POST_OBJECTS_LIKED_BY_THE_PROVIDED_USER_ID,
            ComposioAction.TWITTER_RETURNS_THE_OPEN_API_SPECIFICATION_DOCUMENT,
            ComposioAction.TWITTER_RETURNS_USER_OBJECTS_THAT_ARE_BLOCKED_BY_PROVIDED_USER_ID,
            ComposioAction.TWITTER_RETURNS_USER_OBJECTS_THAT_ARE_MUTED_BY_THE_PROVIDED_USER_ID,
            ComposioAction.TWITTER_RETURNS_USER_OBJECTS_THAT_FOLLOW_A_LIST_BY_THE_PROVIDED_LIST_ID,
            ComposioAction.TWITTER_RETURNS_USER_OBJECTS_THAT_HAVE_LIKED_THE_PROVIDED_POST_ID,
            ComposioAction.TWITTER_RETURNS_USER_OBJECTS_THAT_HAVE_RETWEETED_THE_PROVIDED_POST_ID,
            ComposioAction.TWITTER_SEARCH_FOR_SPACES,
            ComposioAction.TWITTER_SEND_A_NEW_MESSAGE_TO_A_DM_CONVERSATION,
            ComposioAction.TWITTER_SEND_A_NEW_MESSAGE_TO_A_USER,
            ComposioAction.TWITTER_SPACE_LOOKUP_BY_SPACE_ID,
            ComposioAction.TWITTER_SPACE_LOOKUP_BY_THEIR_CREATORS,
            ComposioAction.TWITTER_SPACE_LOOKUP_UP_SPACE_IDS,
            ComposioAction.TWITTER_UNFOLLOW_A_LIST,
            ComposioAction.TWITTER_UNFOLLOW_USER,
            ComposioAction.TWITTER_UNMUTE_USER_BY_USER_ID,
            ComposioAction.TWITTER_UNPIN_A_LIST,
            ComposioAction.TWITTER_UPDATE_LIST,
            ComposioAction.TWITTER_USER_HOME_TIMELINE_BY_USER_ID,
            ComposioAction.TWITTER_USER_LOOKUP_BY_ID,
            ComposioAction.TWITTER_USER_LOOKUP_BY_IDS,
            ComposioAction.TWITTER_USER_LOOKUP_BY_USERNAME,
            ComposioAction.TWITTER_USER_LOOKUP_BY_USERNAMES,
            ComposioAction.TWITTER_USER_LOOKUP_ME,
        ],
    },
}


__all__ = ["COMPOSIO_ACTIONS"]
