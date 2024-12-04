import base64
from io import BytesIO
from os import getenv
from typing import Dict, List
from urllib.parse import quote

import nest_asyncio
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
from phi.tools.streamlit.components import (
    check_password,
    get_openai_key_sidebar,
    get_username_sidebar,
)
from phi.utils.log import logger
from PIL import Image

from llmdj.externalize import activate_django

activate_django()

from dashboard.models import UserConfig

from ai.agents import (
    arxiv,
    base,
    github,
    google_calender,
    journal,
    patent_writer,
    python,
    wikipedia,
    youtube,
)
from ai.coordinators.generic import CoordiantorTeamConfig
from ai.coordinators.generic import get_coordinator as get_generic_leader
from ai.document.reader.excel import ExcelReader
from ai.document.reader.general import GenericReader
from ai.document.reader.image import ImageReader
from ai.document.reader.pptx import PPTXReader
from app.components.popup import show_popup
from app.components.sidebar import create_sidebar

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
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander(":point_down: Examples:"):
    examples = [
        "Write a polite email requesting a meeting with a client.",
        "How does machine learning work?",
        "Generate 5 catchy taglines for a tech startup.",
        "Translate 'Good evening, how are you?' into Spanish.",
        "Explain how an API works in simple terms.",
    ]
    for example in examples:
        st.markdown(
            f"- {example} <a href='?q={quote(example)}' target='_self'>[Try it!]</a>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "Feel free to unleash your creativity and explore the full potential of this platform."
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
    image = Image.open(image_file)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoding}"


def get_selected_assistant_config(session_id, assistant_name):
    label = assistant_name.lower().replace(" ", "_")
    try:
        configs = {
            f"{label}_model_type": "GPT",
            f"{label}_model_id": "gpt-4o",
            f"{label}_temperature": "0",
            f"{label}_agent_enabled": "1",
        }
        ucs = UserConfig.objects.filter(session_id=session_id, key__in=configs.keys())

        for config in ucs:
            configs[config.key] = config.value
            if config.key == f"{label}_agent_enabled":
                configs[config.key] = bool(int(config.value))
            if config.key == f"{label}_temperature":
                configs[config.key] = float(config.value)

        return CoordiantorTeamConfig(
            model=configs[f"{label}_model_type"],
            model_id=configs[f"{label}_model_id"],
            temperature=configs[f"{label}_temperature"],
            enabled=configs[f"{label}_agent_enabled"],
        )
    except ExcelReader as e:
        logger.error("Error reading get_selected_assistant_config(): %s", e)
        return None


def main() -> None:
    # Get OpenAI key
    get_openai_key_sidebar()

    # Get username
    username = "CoPlanet" if getenv("RUNTIME_ENV") == "dev" else get_username_sidebar()
    if username:
        st.sidebar.info(f":technologist: User: {username}")
    else:
        st.markdown("---")
        st.markdown("#### :technologist: Please enter a username")
        return

    # Get Model Id
    model_id = st.sidebar.selectbox("Coordinator", options=["gpt-4o", "gpt-4o-mini"])
    # TODO: add tempature
    # ADD ICON + POPUP
    # EXAMPLES using multi agent 3 times
    # Set model_id in session state
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = model_id
    # Restart the agent if model_id has changed
    elif st.session_state["model_id"] != model_id:
        st.session_state["model_id"] = model_id
        restart_agent()

    # Sidebar checkboxes for selecting team members
    st.sidebar.markdown("### Select Team Members")

    # Initialize session state for popup control
    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

    # Define agents dictionary
    AGENTS = {
        journal.agent_name: journal.get_agent,
        python.agent_name: python.get_agent,
        arxiv.agent_name: arxiv.get_agent,
        youtube.agent_name: youtube.get_agent,
        google_calender.agent_name: google_calender.get_agent,
        github.agent_name: github.get_agent,
        wikipedia.agent_name: wikipedia.get_agent,
        patent_writer.agent_name: patent_writer.get_agent,
    }

    # Get the Agent
    generic_leader: Agent
    SID = (
        st.query_params[SESSION_KEY]
        if (SESSION_KEY in st.query_params and st.query_params[SESSION_KEY])
        else None
    )

    AGENTS_CONFIG: Dict[str, CoordiantorTeamConfig] = {}

    if SID:
        logger.debug(f">>> Using Session ID: {SID}")

        for agent in AGENTS:
            config = get_selected_assistant_config(SID, agent)
            if config:
                AGENTS_CONFIG[agent] = config

        logger.debug("Agents Config: ")
        for agent, config in AGENTS_CONFIG.items():
            logger.debug(f"'{agent}' configed to: {config}")

    if (
        "generic_leader" not in st.session_state
        or st.session_state["generic_leader"] is None
        or (SID and st.session_state["generic_leader"].session_id != SID)
    ):
        logger.debug(f">>> Creating {model_id} Agent")
        generic_leader = get_generic_leader(
            team_config=AGENTS_CONFIG,
            model_id=model_id,
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

    for agent in generic_leader.team:
        agent: base.Agent

        st.session_state[f"{agent.label}_model_id"] = agent.model.id
        st.session_state[f"{agent.label}_model_type"] = agent.model_type
        st.session_state[f"{agent.label}_temperature"] = getattr(
            agent.model, "temperature", 0
        )

    # Create sidebar
    create_sidebar(SID, AGENTS)

    # Show popup if triggered
    if st.session_state.show_popup and st.session_state.selected_assistant:
        show_popup(SID, st.session_state.selected_assistant)
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

    # Display existing chat messages
    for message in st.session_state["messages"]:
        # Skip system and tool messages
        if message.get("role") in ["system", "tool"]:
            continue
        # Display the message
        message_role = message.get("role")
        if message_role is not None:
            with st.chat_message(message_role):
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            st.write(item["text"])
                        elif item["type"] == "image_url":
                            st.image(item["image_url"]["url"], use_column_width=True)
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
                for delta in generic_leader.run(
                    message=question,
                    images=uploaded_images,
                    stream=True,
                ):
                    response += delta.content  # type: ignore
                    resp_container.markdown(response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )

    # Load knowledge base
    if generic_leader.knowledge:
        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0
            st.session_state["input_url"] = ""

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
                                st.session_state["uploaded_images"].append(
                                    auto_rag_documents
                                )
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
                f">>> Loading {model_id} session: {new_generic_leader_session_id}"
            )
            st.query_params[SESSION_KEY] = new_generic_leader_session_id
            st.rerun()
            return
        else:
            logger.debug(
                f">>> Continuing {model_id} session: {new_generic_leader_session_id}"
            )

    NEW_SESSION = st.sidebar.button("New Session")

    if generic_leader.storage:
        if st.sidebar.button("Delete All Session", key="delete_all_session_button"):
            ids = generic_leader.storage.get_all_session_ids()
            for id in ids:
                generic_leader.storage.delete_session(id)
            UserConfig.objects.filter(session_id__in=ids).delete()
            NEW_SESSION = True

    if NEW_SESSION:
        logger.debug(">>> Creating new session...")
        KEYS = list(st.query_params.keys())
        for key in KEYS:
            del st.query_params[key]
        restart_agent()


if check_password():
    main()
