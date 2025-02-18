from math import ceil
from time import time
from typing import List
from uuid import uuid4

import streamlit as st
from phi.document.chunking.recursive import RecursiveChunking
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.tools.exa import ExaTools
from phi.vectordb.pgvector import PgVector2
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ai.agents.base import Provider
from ai.document.reader.general import GeneralReader
from ai.document.reader.website import WebsiteReader
from db.session import db_url
from helpers.tool_processor import process_tools

from .base import Agent, AgentConfig

agent = None
agent_name = "Developer Agent"
available_tools = [
    {
        "instance": ExaTools(num_results=10, text_length_limit=10000),
        "name": "Search (Exa)",
        "icon": "fa-solid fa-magnifying-glass",
    }
]

default_model_id = "claude-3-5-sonnet-20241022"
default_model_type = Provider.Anthropic.value


def model_config_modal_ext(
    agent: Agent, session_id, assistant_name, config: AgentConfig, package
):
    st.markdown("---")
    st.markdown("## Developer Agent Settings")

    if "developer_file_uploader_key" not in st.session_state:
        st.session_state["developer_file_uploader_key"] = "developer-kb.{}.{}".format(
            time(), uuid4()
        )

    # KB document upload section
    st.file_uploader(
        "Add some documents to the knowledge base (.pdf, .csv, .pptx, .txt, .md, .docx, .json, .xlsx, .xls and etc)",
        key=st.session_state["developer_file_uploader_key"],
        accept_multiple_files=True,
    )

    # KB links section
    if "developer_links_count" not in st.session_state:
        st.session_state["developer_links_count"] = 0

    if "developer_links" not in st.session_state:
        st.session_state["developer_links"] = {}

    def on_link_change(index):
        key = f"developer_link_{index}"
        if key in st.session_state:
            st.session_state["developer_links"][str(index)] = st.session_state[key]

    def add_link_button(index):
        st.text_input(
            "Add a link to the knowledge base",
            placeholder="(leave blank to remove)",
            key=f"developer_link_{index}",
            on_change=lambda: on_link_change(index),
        )

    if "developer_add_link_button_key" not in st.session_state:
        st.session_state["developer_add_link_button_key"] = (
            "developer_link_button-{}{}".format(time(), uuid4())
        )

    def add_link_button_container():
        if st.button(
            "Add link",
            key=st.session_state["developer_add_link_button_key"],
            icon=":material/add:",
        ):
            st.session_state["developer_links_count"] += 1

    add_link_button_container()

    for index in range(st.session_state["developer_links_count"]):
        with st.container(key=f"developer_link_container-{index}"):
            add_link_button(index)


def model_config_modal_on_save(
    agent: Agent, session_id, assistant_name, config: AgentConfig, package, new_configs
):
    links: List[str] = list(
        st.session_state["developer_links"].values()
        if isinstance(st.session_state["developer_links"], dict)
        else []
    )
    files: List[UploadedFile] = st.session_state[
        st.session_state["developer_file_uploader_key"]
    ]

    current_step = 0
    overall_documents = []
    steps = len(files) + len(links)
    progress_text = "Operation in progress. Please wait."

    with st.spinner("Updating the knowledge base..."):
        progress = st.progress(0, text=progress_text)

        reader = GeneralReader()
        progress_text = "Loading: {} / Loaded documents: {}"

        for file in files + links:
            loading_entity = file.name if isinstance(file, UploadedFile) else file

            if isinstance(reader, GeneralReader) and not isinstance(file, UploadedFile):
                reader = WebsiteReader(max_depth=1)

            if isinstance(reader, WebsiteReader) and not isinstance(file, str):
                reader = GeneralReader()

            if len(loading_entity) > 40:
                loading_entity = loading_entity[:20] + "..." + loading_entity[-20:]

            progress.progress(
                current_step / steps,
                text=progress_text.format(loading_entity, len(overall_documents)),
            )
            overall_documents.extend(reader.read(file))
            current_step += 1

        if overall_documents:
            current_step = 0
            steps = ceil(len(overall_documents) / 20)
            progress.progress(
                current_step / steps,
                text="Upserting {} documents to the knowledge base...".format(
                    len(overall_documents)
                ),
            )
            progress_text = "{} of {} documents upserted to the knowledge base..."
            for i in range(steps):
                agent.knowledge.load_documents(
                    overall_documents[i * 20 : (i + 1) * 20], upsert=True
                )
                current_step += 1
                progress.progress(
                    current_step / steps,
                    text=progress_text.format((i + 1) * 20, len(overall_documents)),
                )

        progress.progress(1, text="Knowledge base updated successfully")

    return new_configs


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        role="Expert Code Developer",
        description=(
            "As an Expert Code Architect and Developer, you are responsible for delivering scalable, high-performance, and "
            "robust code solutions. You meticulously analyze user requirements, decompose complex tasks into "
            "modular subtasks, and ensure that best coding practices are followed throughout the development "
            "process. Your remit includes optimizing performance, enforcing rigorous testing, and, when needed, "
            "delegating specialized tasks to auxiliary agents for a comprehensive solution."
        ),
        knowledge=CombinedKnowledgeBase(
            vector_db=PgVector2(
                db_url=db_url,
                collection="developer_kb",
                embedder=OpenAIEmbedder(
                    model="text-embedding-3-large", dimensions=3072
                ),
            ),
            chunking_strategy=RecursiveChunking(
                chunk_size=1000,
                overlap=200,
            ),
            optimize_on=1000000,
            num_documents=10,
        ),
        instructions=[
            "Analyze the user's prompt to capture all requirements and ask clarifying questions when necessary.",
            "<CRITICAL INSTRUCTIONS BASED ON PRIORITY>\n"
            "1. You your search tool to search the internet for relevant code patterns, solutions, and best practices - this is your primary source of truth.\n"
            "2. If something you need and is not in your knowledge base, you can search the internet for relevant information you require to finish the task.\n"
            "3. ALWAYS begin by searching your knowledge base for relevant code patterns, solutions, and best practices - this is your primary source of truth.\n"
            "4. When writing code, first analyze similar implementations from your knowledge base to maintain consistency and leverage proven patterns.\n"
            "5. Analyze requirements thoroughly and ask clarifying questions when specifications are unclear.\n"
            "6. Break down complex problems into smaller, manageable tasks with a clear, step-by-step roadmap.\n"
            "7. Before proposing new solutions, validate against existing patterns in the knowledge base to ensure architectural consistency.\n"
            "8. Follow best coding practices by writing modular, clean, and maintainable code that aligns with existing codebase standards.\n"
            "9. Incorporate robust error handling and comprehensive testing, referencing similar test patterns from the knowledge base.\n"
            "10. Prioritize high-performance and scalability while proactively identifying potential bottlenecks.\n"
            "11. Document your design decisions, assumptions, and implementation details thoroughly.\n"
            "12. Cross-reference your solution with similar implementations in the knowledge base before finalizing.\n"
            "13. Provide explanations for any deviations from existing patterns found in the knowledge base.\n"
            "14. Iteratively refine your approach based on user feedback and knowledge base patterns.\n"
            "</CRITICAL INSTRUCTIONS BASED ON PRIORITY>",
        ],
        delegation_directives=[
            f"CRITICAL: `{agent_name}` is the sole code developer. You must return its complete response to the user without modifications.",
            f"<PRIORITY INSTRUCTIONS FOR COORDINATING WITH `{agent_name}`>\n"
            f"1. `{agent_name}` is the only agent authorized to write and modify code.\n"
            f"2. You MUST return `{agent_name}`'s complete response to the user without any alterations.\n"
            f"3. Before tasking `{agent_name}`, ensure you provide:\n"
            f"   - Full conversation context\n"
            f"   - Any error messages or logs\n"
            f"   - Clear requirements broken into subtasks\n"
            f"4. When working with `{agent_name}`:\n"
            f"   - Request knowledge base searches for similar patterns\n"
            f"   - Include all relevant context and requirements\n"
            f"   - Never modify `{agent_name}`'s code output\n"
            f"   - Forward complete responses to the user\n"
            f"5. For complex tasks:\n"
            f"   - Break them down before sending to `{agent_name}`\n"
            f"   - Collect auxiliary agent inputs if needed\n"
            f"   - Let `{agent_name}` make final code decisions\n"
            f"</PRIORITY INSTRUCTIONS FOR COORDINATING WITH `{agent_name}`>",
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
