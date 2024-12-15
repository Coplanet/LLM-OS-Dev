from ai.agents import (
    funny,
    github,
    google_calender,
    journal,
    linkedin_content_generator,
    patent_writer,
    python,
)

ICONS = {
    journal.agent_name: "fa-solid fa-book",
    python.agent_name: "fab fa-python",
    google_calender.agent_name: "fa-solid fa-calendar-alt",
    github.agent_name: "fab fa-github",
    patent_writer.agent_name: "fa-solid fa-lightbulb",
    linkedin_content_generator.agent_name: "fab fa-linkedin",
    funny.agent_name: "fa-solid fa-face-laugh-squint",
}
