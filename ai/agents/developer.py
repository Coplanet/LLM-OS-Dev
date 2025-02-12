import os
from time import time
from typing import List
from uuid import uuid4

import streamlit as st
from phi.document.chunking.recursive import RecursiveChunking
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.reranker.cohere import CohereReranker
from phi.tools.exa import ExaTools
from phi.vectordb.pgvector import PgVector2
from streamlit.runtime.uploaded_file_manager import UploadedFile

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

# default_model = "claude"

# recursive chunking

# develope LLM on:
#  - hybrid search
#  - filtering the KB result
#  - reranker


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
    uploaded_files_ = st.file_uploader(
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

    # render uploaded files and links

    for uploaded_file in uploaded_files_:
        document_name = uploaded_file.name
        st.write(document_name)

    for link in st.session_state["developer_links"].values():
        st.write(link)


def model_config_modal_on_save(
    agent: Agent, session_id, assistant_name, config: AgentConfig, package, new_configs
):
    links: List[str] = (
        st.session_state["developer_links"]
        if isinstance(st.session_state["developer_links"], dict)
        else []
    )
    files: List[UploadedFile] = st.session_state[
        st.session_state["developer_file_uploader_key"]
    ]

    current_step = 0
    overall_documents = []
    steps = len(files) + len(links) + 1
    progress_text = "Operation in progress. Please wait."

    with st.spinner("Uploading knowledge base..."):
        progress = st.progress(0, text=progress_text)

        reader = GeneralReader()
        progress_text = "Loading: {} / Loaded documents: {}"

        for file in files + links:
            loading_entity = file.name if isinstance(file, UploadedFile) else file

            if isinstance(reader, GeneralReader) and not isinstance(file, UploadedFile):
                reader = WebsiteReader(max_depth=1)

            if isinstance(reader, WebsiteReader) and not isinstance(file, str):
                reader = GeneralReader()

            if len(loading_entity) > 100:
                loading_entity = loading_entity[:50] + "..." + loading_entity[-50:]

            progress.progress(
                current_step * 100 / steps,
                text=progress_text.format(loading_entity, len(overall_documents)),
            )
            overall_documents.extend(reader.read(file))
            current_step += 1

        progress_text = "Upserting {} documents to the knowledge base..."
        progress.progress(
            current_step * 100 / steps,
            text=progress_text.format(len(overall_documents)),
        )

        if overall_documents:
            agent.knowledge.load_documents(overall_documents, upsert=True)

        progress.progress(100, text="Knowledge base updated successfully")

    return new_configs


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    vectordb_kwargs = {
        "db_url": db_url,
        "collection": "developer_kb",
        "embedder": OpenAIEmbedder(model="text-embedding-3-large", dimensions=3072),
    }

    if COHERE_API_KEY:
        vectordb_kwargs["reranker"] = CohereReranker(
            model="rerank-multilingual-v3.0", api_key=COHERE_API_KEY
        )

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
            vector_db=PgVector2(**vectordb_kwargs),
            chunking_strategy=RecursiveChunking(
                chunk_size=1000,
                chunk_overlap=200,
            ),
            optimize_on=1000000,
            num_documents=5,
        ),
        instructions=[
            "Analyze the user's prompt to capture all requirements and ask clarifying questions when needed.",
            "Leverage your internal knowledge base and previous experiences to guide solution design.",
            "Follow best coding practices by writing modular, clean, and maintainable code.",
            "Break down complex problems into smaller, manageable tasks with a clear roadmap.",
            "Prioritize high-performance and scalability while recognizing potential bottlenecks.",
            "Incorporate robust error handling and comprehensive testing to ensure code reliability.",
            "Document your design decisions, assumptions, and implementation details thoroughly.",
            "Iterate and refine your approach based on user feedback and testing outcomes.",
        ],
        delegation_directives=[
            (
                "You must supply the complete execution context including conversation history, current "
                f"error/log messages, and performance metrics to ensure full situational awareness to `{agent_name}`."
            ),
            (
                "Before passing tasks to the agent, preprocess and format all incoming requirements to segment them into "
                f"actionable items; include any necessary sub-task details to `{agent_name}`."
            ),
            (
                "Where applicable, delegate off-scope or highly specialized tasks to auxiliary agents and collate their "
                f"results for the primary agent to integrate to `{agent_name}`."
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
