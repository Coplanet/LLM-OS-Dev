import base64
import re
from io import BytesIO
from os import getenv
from time import time
from typing import Dict, List, Optional
from urllib.parse import quote

import nest_asyncio
import sqlalchemy as sql
import streamlit as st
from phi.agent import Agent
from phi.document import Document
from phi.document.reader import Reader
from phi.document.reader.csv_reader import CSVReader
from phi.document.reader.docx import DocxReader
from phi.document.reader.json import JSONReader
from phi.document.reader.pdf import PDFReader
from phi.document.reader.text import TextReader
from phi.document.reader.website import WebsiteReader
from phi.model.content import Image
from phi.model.message import Message
from phi.tools.streamlit.components import (
    check_password,
    get_openai_key_sidebar,
    get_username_sidebar,
)
from PIL import Image as PILImage

from ai.agents import (
    base,
    funny,
    journal,
    linkedin_content_generator,
    patent_writer,
    python,
    settings,
)
from ai.agents.settings import agent_settings
from ai.coordinators import generic as coordinator
from ai.document.reader.excel import ExcelReader
from ai.document.reader.general import GenericReader
from ai.document.reader.image import ImageReader
from ai.document.reader.pptx import PPTXReader
from app.components.popup import show_popup
from app.components.sidebar import create_sidebar
from app.utils import to_label
from db.session import get_db_context
from db.tables import UserConfig
from helpers.log import logger

STATIC_DIR = "app/static"
IMAGE_DIR = f"{STATIC_DIR}/images"

SESSION_KEY = "sid"

nest_asyncio.apply()
st.set_page_config(page_title="CoPlanet AI", page_icon=f"{IMAGE_DIR}/favicon.png")

st.title("CoPlanet AI")
st.markdown(
    f"""\
    ##### <img src="{IMAGE_DIR}/coplanet.png" alt="Logo" style="width: 30px; margin-right: 10px;"> \
    Unleashing Infinite Possibilities: Where Technology Meets Community\
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.0/css/all.min.css">""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
        /* Target the sidebar container */
        [data-testid="stSidebarContent"] {
            overflow-x: hidden;
        }
        .st-key-delete_knowledge_base button,
        .st-key-delete_all_session_button button {
            color: #fff;
            background-color: #dc3545;
            border-color: #dc3545;
        }
        .st-key-delete_knowledge_base button:hover,
        .st-key-delete_all_session_button button:hover {
            color: #fff;
            background-color: #c82333;
            border-color: #bd2130;
        }
        .row-widget.stCheckbox {
            min-height: unset !important;
        }
        .stColumn {
            overflow-x: hidden;
        }
        .stVerticalBlock {
            gap: 0.75rem !important;
        }
        .stHorizontalBlock {
            gap: 0.5rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def restart_agent():
    logger.debug(">>> Restarting Agent")
    st.session_state["generic_leader"] = None
    st.session_state["generic_leader_session_id"] = None
    st.session_state["uploaded_image"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    if "image_uploader_key" in st.session_state:
        st.session_state["image_uploader_key"] += 1
    st.rerun()


def encode_image(image_file):
    image = PILImage.open(image_file)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoding}"


def get_selected_assistant_config(session_id, label, package):
    try:
        available_tools_manifest = {}
        if isinstance(package.available_tools, dict):
            raise ValueError("Available tools must be a list of dictionaries")

        for tool in package.available_tools:
            if not isinstance(tool, dict):
                raise ValueError(
                    f"Tool '{tool.__name__}' is not a dictionary in package '{package.agent_name}'."
                )
            name = tool.get("name", None)
            key = name

            if "group" in tool:
                key = f"group:{tool['group']}"
            if key not in available_tools_manifest:
                available_tools_manifest[key] = []
            available_tools_manifest[key].append(tool)

        default_configs = {
            "model_type": "OpenAI",
            "model_id": "gpt-4o",
            "temperature": 0,
            "enabled": True,
            "max_tokens": agent_settings.default_max_completion_tokens,
            "tools": {k: True for k in available_tools_manifest.keys()},
        }

        with get_db_context() as db:
            config: UserConfig = UserConfig.get_models_config(
                db, session_id, auto_create=False
            )

        if config:
            if isinstance(config.value_json, dict) and label in config.value_json:
                for key in default_configs:
                    if key in config.value_json[label]:
                        default_configs[key] = config.value_json[label][key]
        tools = {}
        for key, subtools in available_tools_manifest.items():
            if key in default_configs["tools"]:
                for tool in subtools:
                    tools[tool.get("name")] = tool

        return settings.AgentConfig(
            provider=default_configs["model_type"],
            model_id=default_configs["model_id"],
            temperature=default_configs["temperature"],
            enabled=default_configs["enabled"],
            max_tokens=default_configs["max_tokens"],
            tools=tools,
        )
    except Exception as e:
        logger.error("Error reading get_selected_assistant_config(): %s", e)
        return None


def main() -> None:
    # Get OpenAI key
    get_openai_key_sidebar()

    # Get username
    username = "CoPlanet" if getenv("RUNTIME_ENV") == "dev" else get_username_sidebar()
    if username:
        with st.expander(":point_down: Examples:"):
            examples = [
                "Report on the latest AI technology updates at your choice from YouTube channels.",
                "Identify the best AI framework on GitHub and star it upon confirmation.",
                "Summarize a Wikipedia page and YouTube video on quantum computers.",
            ]
            for example in examples:
                st.markdown(
                    f"- {example} <a href='?q={quote(example)}' target='_self'>[Try it!]</a>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "Feel free to unleash your creativity and explore the full potential of this platform."
            )

        st.sidebar.info(f":technologist: User: {username}")
    else:
        st.markdown("---")
        st.markdown("#### :technologist: Please enter a username")
        return

    # Initialize session state for popup control
    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

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

    # Get the Agent
    generic_leader: Agent
    SID = (
        st.query_params[SESSION_KEY]
        if (SESSION_KEY in st.query_params and st.query_params[SESSION_KEY])
        else None
    )

    LEADER_CONFIG = settings.AgentConfig.empty()
    AGENTS_CONFIG: Dict[str, settings.AgentConfig] = {}

    if SID:
        logger.debug(f">>> Using Session ID: {SID}")

        for agent, agent_config in AGENTS.items():
            config = get_selected_assistant_config(
                SID, agent_config.get("label", agent), agent_config.get("package")
            )
            if agent_config.get("is_leader", False):
                LEADER_CONFIG = config
            else:
                AGENTS_CONFIG[agent] = config

        logger.debug("Agents Config: ")
        for agent, config in AGENTS_CONFIG.items():
            logger.debug(f"'{agent}' configed to: {config}")

    if (
        "generic_leader" not in st.session_state
        or st.session_state["generic_leader"] is None
        or (SID and st.session_state["generic_leader"].session_id != SID)
    ):
        logger.debug(">>> Creating leader agent with config: %s", LEADER_CONFIG)
        coordinator.agent = generic_leader = coordinator.get_coordinator(
            team_config=AGENTS_CONFIG,
            config=LEADER_CONFIG,
            session_id=SID,
        )
        st.session_state["generic_leader"] = generic_leader
    else:
        generic_leader = st.session_state["generic_leader"]

    NEW_SESSION = not bool(SID)
    # Create Agent session (i.e. log to database) and save session_id in session state
    try:
        if SID:
            st.session_state["generic_leader_session_id"] = st.query_params[SESSION_KEY]
            logger.debug(">>> Identified Session ID: %s", SID)
        else:
            st.query_params[SESSION_KEY] = generic_leader.create_session()
            st.rerun()
            return

    except Exception as e:
        logger.error(e)
        st.warning("Could not create Agent session, is the database running?")
        return

    for agent in [generic_leader] + generic_leader.team:
        agent: base.Agent

        config: settings.AgentConfig = (
            AGENTS_CONFIG.get(agent.name) or settings.AgentConfig.empty()
        )

        st.session_state[f"{agent.label}_model_id"] = config.model_id or agent.model.id
        st.session_state[f"{agent.label}_model_type"] = (
            config.provider or agent.model_type
        )
        st.session_state[f"{agent.label}_temperature"] = config.temperature or getattr(
            agent.model, "temperature", 0
        )
        st.session_state[f"{agent.label}_max_tokens"] = config.max_tokens or getattr(
            agent.model, "max_tokens", agent_settings.default_max_completion_tokens
        )

    # Create sidebar
    create_sidebar(SID, {generic_leader.name: AGENTS[generic_leader.name]})

    # Sidebar checkboxes for selecting team members
    st.sidebar.markdown("### Select Team Members")
    create_sidebar(
        SID, {k: v for k, v in AGENTS.items() if not v.get("is_leader", False)}
    )

    # Show popup if triggered
    if st.session_state.show_popup and st.session_state.selected_assistant:
        selected_assistant = st.session_state.selected_assistant
        agent = AGENTS[selected_assistant]
        package = agent.get("package")
        agent_config = (
            AGENTS_CONFIG[package.agent_name]
            if not agent.get("is_leader", False)
            else LEADER_CONFIG
        )
        show_popup(SID, selected_assistant, agent_config, package)
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

    # Store uploaded image in session state
    if "uploaded_images" not in st.session_state:
        st.session_state["uploaded_images"] = []

    uploaded_images = st.session_state["uploaded_images"]

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # Load existing messages
    agent_chat_history = generic_leader.memory.get_messages()
    if not agent_chat_history:
        history = generic_leader.read_from_storage()
        if (
            history
            and history.memory
            and "chats" in history.memory
            and isinstance(history.memory["chats"], list)
        ):
            for chat in history.memory["chats"]:
                if "messages" in chat:
                    if "role" in chat["message"] and chat["message"]["role"] == "user":
                        agent_chat_history.append(
                            {"role": "user", "content": chat["messages"]["content"]}
                        )
                        if "response" in chat:
                            agent_chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": chat["response"]["content"],
                                }
                            )

    if agent_chat_history:
        logger.debug("Loading chat history")
        null_content = []
        for i in range(len(agent_chat_history)):
            if agent_chat_history[i].get("content") is None:
                null_content.append(i)
            elif isinstance(agent_chat_history[i]["content"], str):
                agent_chat_history[i]["content"] = re.sub(
                    r"[\n\s]*!\[[^\]]+?\]\([^\)]+?\)",
                    "",
                    agent_chat_history[i]["content"],
                )
        # remove null content using the indices:
        for i in reversed(null_content):
            agent_chat_history.pop(i)
        st.session_state["messages"] = agent_chat_history

    else:
        if "q" in st.query_params:
            st.session_state["messages"] = [
                {"role": "user", "content": st.query_params["q"]}
            ]
        else:
            logger.debug("No chat history found")
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Ask me anything..."}
            ]

    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    last_user_message_container = None
    user_last_message_image_render = False
    # Display existing chat messages
    for message in st.session_state["messages"]:
        # Skip system and tool messages
        if message.get("role") in ["system", "tool"]:
            continue
        # Display the message
        message_role = message.get("role")
        if message_role is not None:
            chat_message_container = st.chat_message(message_role)
            if message_role == "user":
                last_user_message_container = chat_message_container
            else:
                last_user_message_container = None
            with chat_message_container:
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            st.write(item["text"])
                        elif item["type"] == "image_url":
                            st.image(
                                item["image_url"]["url"],
                                caption=item.get("image_caption"),
                                use_column_width=True,
                            )
                            if message_role == "user":
                                user_last_message_image_render = True
                            else:
                                user_last_message_image_render = False
                else:
                    st.write(content)

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp_container = st.empty()
                response = ""
                start = time()
                for delta in generic_leader.run(
                    message=question,
                    images=uploaded_images,
                    stream=True,
                ):
                    response += delta.content  # type: ignore
                    response = re.sub(r"[\n\s]*!\[[^\]]+?\]\([^\)]+?\)", "", response)
                    resp_container.markdown(response)
                end = time()
                logger.debug("Time to response: {:.2f} seconds".format(end - start))

            # Get the images
            image_outputs: Optional[List[Image]] = generic_leader.get_images()

            if uploaded_images and not user_last_message_image_render:
                memory = generic_leader.read_from_storage().memory
                if (
                    memory
                    and "messages" in memory
                    and isinstance(memory["messages"], list)
                    and len(memory["messages"]) > 1
                ):
                    previous_message = memory["messages"][-2]
                    if (
                        previous_message.get("role") == "user"
                        and previous_message.get("images")
                        and isinstance(previous_message["images"], list)
                        and previous_message["images"]
                    ):
                        with last_user_message_container:
                            for img in previous_message["images"]:
                                st.image(img)

            # Render the images
            if image_outputs:
                logger.debug("Rendering '{}' images...".format(len(image_outputs)))
                contents = []
                contents.append({"type": "text", "text": response})

                for img in image_outputs:
                    if isinstance(img, dict):
                        img = Image.model_validate(img)

                    if not isinstance(img, Image):
                        logger.error(
                            "Image is not a valid Image model; it is a: {} [skipping]".format(
                                type(img)
                            )
                        )
                        continue

                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img.url},
                            "image_caption": img.original_prompt,
                        }
                    )
                    st.image(img.url, caption=img.original_prompt)

                generic_leader.images = []

                st.session_state["messages"].append(
                    {"role": "assistant", "content": contents}
                )
                generic_leader.memory.add_message(
                    Message(role="assistant", content=contents[1:])
                )
                generic_leader.write_to_storage()

            else:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

    # Load knowledge base
    if generic_leader.knowledge:
        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0
            st.session_state["input_url"] = ""

        # TODO: Improve this section digestion speed
        if False:
            input_url = st.sidebar.text_input(
                "Add URL to Knowledge Base",
                type="default",
                key=st.session_state["url_scrape_key"],
            )
            add_url_button = st.sidebar.button("Add URL")
            if add_url_button:
                if input_url is not None:
                    alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                    if f"{input_url}_scraped" not in st.session_state:
                        scraper = WebsiteReader(max_links=10, max_depth=3)
                        web_documents: List[Document] = scraper.read(input_url)
                        if web_documents:
                            generic_leader.knowledge.load_documents(
                                web_documents, upsert=True
                            )
                        else:
                            st.sidebar.error("Could not read website")
                        st.session_state[f"{input_url}_uploaded"] = True
                    alert.empty()

        # -*- Add documents to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100
        uploaded_files_ = st.sidebar.file_uploader(
            "Add a Document (image files, .pdf, .csv, .pptx, .txt, .md, .docx, .json, .xlsx, .xls and etc)",
            key=st.session_state["file_uploader_key"],
            accept_multiple_files=True,
        )
        if uploaded_files_:
            alert = st.sidebar.info(
                "Processing {} document{}...".format(
                    len(uploaded_files_), "s" if len(uploaded_files_) > 1 else ""
                ),
                icon="🧠",
            )
            uploaded_images = []
            for uploaded_file in uploaded_files_:
                document_name = uploaded_file.name
                if f"{document_name}_uploaded" not in st.session_state:
                    file_type = uploaded_file.name.split(".")[-1].lower()

                    reader: Reader
                    if file_type == "pdf":
                        reader = PDFReader()
                    elif file_type == "csv":
                        reader = CSVReader()
                    elif file_type == "pptx":
                        reader = PPTXReader()
                    elif file_type in ["txt", "md"]:
                        reader = TextReader()
                    elif file_type == "docx":
                        reader = DocxReader()
                    elif file_type == "json":
                        reader = JSONReader()
                    elif file_type in ["xlsx", "xls"]:
                        reader = ExcelReader()
                    elif file_type in [
                        "png",
                        "jpg",
                        "jpeg",
                        "gif",
                        "bmp",
                        "tiff",
                        "webp",
                    ]:
                        reader = ImageReader()
                    else:
                        reader = GenericReader()

                    try:
                        auto_rag_documents: List[Document] = reader.read(uploaded_file)

                        if auto_rag_documents:
                            if not isinstance(reader, ImageReader):
                                generic_leader.knowledge.load_documents(
                                    auto_rag_documents, upsert=True
                                )
                                st.session_state["uploaded_files"].append(document_name)
                            else:
                                for image in auto_rag_documents:
                                    uploaded_images.append(image.content)
                        else:
                            st.sidebar.error(
                                "Could not read document: {}".format(document_name)
                            )
                        st.session_state[f"{document_name}_uploaded"] = True
                        st.sidebar.success(
                            "Document: `{}` successfully added to knowledge base.".format(
                                document_name
                            )
                        )

                    except Exception as e:
                        logger.error(e)
                        st.sidebar.error(
                            "Could not read document: {}".format(document_name)
                        )
                        continue
            if uploaded_images:
                st.session_state["uploaded_images"] = uploaded_images
            st.session_state["file_uploader_key"] += 1
            alert.empty()

        if generic_leader.knowledge.vector_db:
            if st.sidebar.button("Delete Knowledge Base", key="delete_knowledge_base"):
                generic_leader.knowledge.vector_db.delete()
                st.sidebar.success("Knowledge base deleted")

    if generic_leader.storage:
        selectbox_index = 0
        generic_leader_session_ids: List[str] = (
            generic_leader.storage.get_all_session_ids()
        )
        for index, id in enumerate(generic_leader_session_ids):
            if id == SID:
                selectbox_index = index
                break

        new_generic_leader_session_id = st.sidebar.selectbox(
            "Session ID", options=generic_leader_session_ids, index=selectbox_index
        )
        if SID != new_generic_leader_session_id:
            logger.debug(
                f">>> Loading {generic_leader.model.id} session: {new_generic_leader_session_id}"
            )
            st.query_params[SESSION_KEY] = new_generic_leader_session_id
            st.rerun()
            return
        else:
            logger.debug(
                f">>> Continuing {generic_leader.model.id} session: {new_generic_leader_session_id}"
            )

    NEW_SESSION = st.sidebar.button("New Session")

    if generic_leader.storage:
        if st.sidebar.button("Delete All Session", key="delete_all_session_button"):
            ids = generic_leader.storage.get_all_session_ids()
            for id in ids:
                generic_leader.storage.delete_session(id)
            if ids:
                with get_db_context() as db:
                    db.execute(
                        sql.delete(UserConfig).where(UserConfig.session_id.in_(ids))
                    )
                    db.commit()
            NEW_SESSION = True

    if NEW_SESSION:
        logger.debug(">>> Creating new session...")
        KEYS = list(st.query_params.keys())
        for key in KEYS:
            del st.query_params[key]
        restart_agent()


if check_password():
    main()
