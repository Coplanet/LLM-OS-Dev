from ai.agents import funny, journal, linkedin_content_generator, patent_writer, python
from ai.coordinators import generic as coordinator
from app.utils import to_label

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

__all__ = ["AGENTS"]
