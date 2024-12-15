from ai.agents import (
    arxiv,
    funny,
    github,
    google_calender,
    journal,
    linkedin_content_generator,
    patent_writer,
    python,
    wikipedia,
)

ICONS = {
    journal.agent_name: "fa-solid fa-book",
    python.agent_name: "fab fa-python",
    arxiv.agent_name: "fa-solid fa-book-open",
    google_calender.agent_name: "fa-solid fa-calendar-alt",
    github.agent_name: "fab fa-github",
    wikipedia.agent_name: "fab fa-wikipedia-w",
    patent_writer.agent_name: "fa-solid fa-lightbulb",
    linkedin_content_generator.agent_name: "fab fa-linkedin",
    funny.agent_name: "fa-solid fa-face-laugh-squint",
}
