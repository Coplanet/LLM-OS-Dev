from ai.agents import (
    arxiv,
    github,
    google_calender,
    journal,
    patent_writer,
    python,
    wikipedia,
    youtube,
)

ICONS = {
    "calculator_enabled": "fa-solid fa-calculator",
    "file_tools_enabled": "fa-solid fa-file",
    "resend_tools_enabled": "fa-solid fa-paper-plane",
    "ddg_search_enabled": "fa-solid fa-search",
    "finance_tools_enabled": "fa-solid fa-chart-line",
    journal.agent_name: "fa-solid fa-book",
    python.agent_name: "fab fa-python",
    arxiv.agent_name: "fa-solid fa-book-open",
    youtube.agent_name: "fab fa-youtube",
    google_calender.agent_name: "fa-solid fa-calendar-alt",
    github.agent_name: "fab fa-github",
    wikipedia.agent_name: "fab fa-wikipedia-w",
    patent_writer.agent_name: "fa-solid fa-lightbulb",
}
