from ai.agents import (
    funny,
    github,
    gmail,
    google_calendar,
    journal,
    linkedin_content_generator,
    patent_writer,
    python,
    tweeter,
)
from ai.coordinators import generic as coordinator
from app.auth import User
from app.components.composio_integrations import AVAILABLE_APPS, App
from app.utils import to_label
from db.tables.user_config import UserIntegration

# Define agents dictionary
AGENTS = {
    coordinator.agent_name: {
        "icon": "fa-solid fa-sitemap",
        "selectable": False,
        "is_leader": True,
        "label": to_label(coordinator.agent_name),
        "get_agent": coordinator.get_coordinator,
        "package": coordinator,
    },
    journal.agent_name: {
        "label": to_label(journal.agent_name),
        "get_agent": journal.get_agent,
        "package": journal,
    },
    python.agent_name: {
        "label": to_label(python.agent_name),
        "get_agent": python.get_agent,
        "package": python,
    },
    patent_writer.agent_name: {
        "label": to_label(patent_writer.agent_name),
        "get_agent": patent_writer.get_agent,
        "package": patent_writer,
    },
    linkedin_content_generator.agent_name: {
        "label": to_label(linkedin_content_generator.agent_name),
        "get_agent": linkedin_content_generator.get_agent,
        "package": linkedin_content_generator,
    },
    funny.agent_name: {
        "label": to_label(funny.agent_name),
        "get_agent": funny.get_agent,
        "package": funny,
    },
}

COMPOSIO_AGENTS = {
    App.TWITTER: {
        "label": to_label(tweeter.agent_name),
        "get_agent": tweeter.get_agent,
        "package": tweeter,
    },
    App.GITHUB: {
        "label": to_label(github.agent_name),
        "get_agent": github.get_agent,
        "package": github,
    },
    App.GOOGLECALENDAR: {
        "label": to_label(google_calendar.agent_name),
        "get_agent": google_calendar.get_agent,
        "package": google_calendar,
    },
    App.GMAIL: {
        "label": to_label(gmail.agent_name),
        "get_agent": gmail.get_agent,
        "package": gmail,
    },
}


def get_available_agents(user: User) -> dict:
    agents = AGENTS.copy()

    for i in UserIntegration.get_integrations(user.user_id):
        app = App(i.app)
        if app in COMPOSIO_AGENTS and app in AVAILABLE_APPS:
            agents[COMPOSIO_AGENTS[app]["package"].agent_name] = COMPOSIO_AGENTS[app]

    return agents


__all__ = ["AGENTS", "get_available_agents"]
