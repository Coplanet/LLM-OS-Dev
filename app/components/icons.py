from ai.agents import (
    developer,
    funny,
    github,
    gmail,
    google_calendar,
    journal,
    linkedin_content_generator,
    patent_writer,
    reasoning,
    task_generator,
    twitter,
)

ICONS = {
    journal.agent_name: "fa-solid fa-book",
    patent_writer.agent_name: "fa-solid fa-lightbulb",
    linkedin_content_generator.agent_name: "fab fa-linkedin",
    funny.agent_name: "fa-solid fa-face-laugh-squint",
    twitter.agent_name: "fab fa-twitter",
    github.agent_name: "fab fa-github",
    google_calendar.agent_name: "fa-solid fa-calendar-days",
    gmail.agent_name: "fa-solid fa-envelope",
    reasoning.agent_name: "fa-solid fa-brain",
    task_generator.agent_name: "fa-brands fa-jira",
    developer.agent_name: "fa-solid fa-code",
}
