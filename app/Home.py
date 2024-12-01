import base64
from io import BytesIO
from os import getenv
from typing import List
from urllib.parse import quote

import nest_asyncio
import streamlit as st
from phi.agent import Agent
from phi.document import Document
from phi.document.reader import Reader
from phi.document.reader.csv_reader import CSVReader
from phi.document.reader.docx import DocxReader
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

from ai.coordinators.generic import get_leader as get_generic_leader

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

with st.expander(":point_down: How to Utilize This Platform"):
    examples = [
        (
            "Enrich your knowledge base by adding insightful blog posts. For example, explore insights \
                from [Sam Altman's blog](https://blog.samaltman.com/what-i-wish-someone-had-told-me) and inquire: "
            "What are the key lessons Sam Altman wished he had known?"
        ),
        "Discover the capabilities of your AI agents by asking: What functions can my agents perform?",
        "Explore global events with our Web Search feature: What's the latest news from France?",
        "Leverage our Calculator for complex computations: How do you calculate 10 factorial?",
        "Stay informed on financial markets: What's the current stock price of AAPL?",
        "Conduct in-depth financial analyses: Compare NVIDIA and AMD using all available financial tools, \
            and distill the essential insights.",
        "Dive into research: Generate a comprehensive report on the HashiCorp and IBM acquisition.",
    ]
    SID = (
        "{}={}".format(SESSION_KEY, st.query_params[SESSION_KEY])
        if SESSION_KEY in st.query_params and st.query_params[SESSION_KEY]
        else ""
    )
    for example in examples:
        st.markdown(
            f"- {example} <a href='?{SID}&q={quote(example)}' target='_self'>[Try it!]</a>",
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


ICONS = {
    "calculator_enabled": "fa-solid fa-calculator",
    "file_tools_enabled": "fa-solid fa-file",
    "resend_tools_enabled": "fa-solid fa-paper-plane",
    "ddg_search_enabled": "fa-solid fa-search",
    "finance_tools_enabled": "fa-solid fa-chart-line",
    "journal_assistant_enabled": "fa-solid fa-book",
    "python_assistant_enabled": "fab fa-python",
    "arxiv_assistant_enabled": "fa-solid fa-book-open",
    "youtube_assistant_enabled": "fab fa-youtube",
    "google_calender_assistant_enabled": "fa-solid fa-calendar-alt",
    "github_assistant_enabled": "fab fa-github",
    "wikipedia_assistant_enabled": "fab fa-wikipedia-w",
    "patent_writer_assistant_enabled": "fa-solid fa-lightbulb",
}


def enable_feature(feature_name: str, checkbox_text: str, help: str) -> bool:
    # Enable Calculator
    if feature_name not in st.session_state:
        st.session_state[feature_name] = True
    # fetch the icon for the feature
    ICON = ICONS.get(feature_name, "")
    # Get calculator_enabled from session state if set
    pre_value = st.session_state[feature_name]
    # Inline layout for checkbox with icon and label
    with st.sidebar:
        col1, col2 = st.columns([1, 11])
        with col1:
            # Create the checkbox with a unique key
            new_value = st.checkbox(
                "label",
                value=pre_value,
                key=f"checkbox_{feature_name}",
                label_visibility="collapsed" if ICON else "visible",
            )

        if ICON:
            with col2:
                # Create an HTML label tied to the checkbox
                st.markdown(
                    f"""
                    <span style="position: relative">
                        <span style="top: 27px; position: absolute; width: 100vw;">
                            <i class="{ICON}" style="margin-right: 10px;"></i> {checkbox_text}
                        </span>
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

    if pre_value != new_value:
        print(st.session_state[feature_name], new_value)
        st.session_state[feature_name] = new_value
        pre_value = new_value
        restart_agent()

    return new_value


def main() -> None:
    # Get OpenAI key from environment variable or user input
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
    model_id = st.sidebar.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"])
    # Set model_id in session state
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = model_id
    # Restart the agent if model_id has changed
    elif st.session_state["model_id"] != model_id:
        st.session_state["model_id"] = model_id
        restart_agent()

    # Sidebar checkboxes for selecting team members
    st.sidebar.markdown("### Select Team Members")
    journal_assistant_enabled = enable_feature(
        "journal_assistant_enabled",
        "Journal Assistant",
        "Enable the journal assistant (uses Exa).",
    )
    python_assistant_enabled = enable_feature(
        "python_assistant_enabled",
        "Python Assistant",
        "Enable the Python Assistant for writing and running python code.",
    )
    arxiv_assistant_enabled = enable_feature(
        "arxiv_assistant_enabled",
        "Arxiv Assistant",
        "Enable the Arxiv Assistant for searching papers in arxiv.",
    )
    youtube_assistant_enabled = enable_feature(
        "youtube_assistant_enabled",
        "Youtube Assistant",
        "Enable the Youtube Assistant for youtube related URLs and Queries.",
    )
    google_calender_assistant_enabled = enable_feature(
        "google_calender_assistant_enabled",
        "Google Calendar Assistant",
        "Enable the Google Assistant",
    )
    github_assistant_enabled = enable_feature(
        "github_assistant_enabled", "GitHub Assistant", "Enable the Google Assistant"
    )
    wikipedia_assistant_enabled = enable_feature(
        "wikipedia_assistant_enabled",
        "Wikipedia Assistant",
        "Enable the Wikipedia Assistant for youtube related URLs and Queries.",
    )
    patent_writer_assistant_enabled = enable_feature(
        "patent_writer_assistant_enabled",
        "Patent Writer Assistant",
        "Enable the Patent Assistant for writing patents",
    )

    # Get the Agent
    generic_leader: Agent
    SID = (
        st.query_params[SESSION_KEY]
        if (SESSION_KEY in st.query_params and st.query_params[SESSION_KEY])
        else None
    )
    if SID:
        logger.info(f">>> Using Session ID: {SID}")
    if (
        "generic_leader" not in st.session_state
        or st.session_state["generic_leader"] is None
        or (SID and st.session_state["generic_leader"].session_id != SID)
    ):
        logger.info(f">>> Creating {model_id} Agent")
        generic_leader = get_generic_leader(
            python_assistant=python_assistant_enabled,
            youtube_assistant=youtube_assistant_enabled,
            arxiv_assistant=arxiv_assistant_enabled,
            journal_assistant=journal_assistant_enabled,
            wikipedia_assistant=wikipedia_assistant_enabled,
            github_assistant=github_assistant_enabled,
            google_calender_assistant=google_calender_assistant_enabled,
            patent_writer_assistant=patent_writer_assistant_enabled,
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
            logger.info(">>> Identified Session ID: %s", st.query_params[SESSION_KEY])
        else:
            st.query_params[SESSION_KEY] = generic_leader.create_session()
            st.rerun()
            return

    except Exception as e:
        logger.error(e)
        st.warning("Could not create Agent session, is the database running?")
        return

    # Store uploaded image in session state
    uploaded_image = None
    if "uploaded_image" in st.session_state:
        uploaded_image = st.session_state["uploaded_image"]

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
        # Search for uploaded image
        if False:
            if uploaded_image is None:
                for message in agent_chat_history:
                    if message.get("role") == "user":
                        content = message.get("content")
                        if isinstance(content, list):
                            for item in content:
                                if item["type"] == "image_url":
                                    uploaded_image = item["image_url"]["url"]
                                    st.session_state["uploaded_image"] = uploaded_image
                                    break
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

    if False:
        # Upload Image
        if uploaded_image is None:
            if "image_uploader_key" not in st.session_state:
                st.session_state["image_uploader_key"] = 200
            uploaded_file = st.sidebar.file_uploader(
                "Upload Image",
                key=st.session_state["image_uploader_key"],
            )
            if uploaded_file is not None:
                alert = st.sidebar.info("Processing Image...", icon="ℹ️")
                image_file_name = uploaded_file.name.split(".")[0]
                if f"{image_file_name}_uploaded" not in st.session_state:
                    logger.info(f"Encoding {image_file_name}")
                    uploaded_image = encode_image(uploaded_file)
                    st.session_state["uploaded_image"] = uploaded_image
                    st.session_state[f"{image_file_name}_uploaded"] = True
                alert.empty()

        # Prompt for user input
        if uploaded_image:
            with st.expander("Uploaded Image", expanded=False):
                st.image(uploaded_image, use_column_width=True)
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
                    images=[uploaded_image] if uploaded_image else [],
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
        uploaded_file = st.sidebar.file_uploader(
            "Add a Document (.pdf)",
            type=["pdf"],
            key=st.session_state["file_uploader_key"],
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing document...", icon="🧠")
            document_name = uploaded_file.name.split(".")[0]
            if f"{document_name}_uploaded" not in st.session_state:
                file_type = uploaded_file.name.split(".")[-1].lower()

                reader: Reader
                if file_type == "pdf":
                    reader = PDFReader()
                elif False and file_type == "csv":
                    reader = CSVReader()
                elif False and file_type == "txt":
                    reader = TextReader()
                elif False and file_type == "docx":
                    reader = DocxReader()
                auto_rag_documents: List[Document] = reader.read(uploaded_file)
                if auto_rag_documents:
                    generic_leader.knowledge.load_documents(
                        auto_rag_documents, upsert=True
                    )
                else:
                    st.sidebar.error("Could not read document")
                st.session_state[f"{document_name}_uploaded"] = True
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
            logger.info(
                f">>> Loading {model_id} session: {new_generic_leader_session_id}"
            )
            st.query_params[SESSION_KEY] = new_generic_leader_session_id
            st.rerun()
            return
        else:
            logger.info(
                f">>> Continuing {model_id} session: {new_generic_leader_session_id}"
            )

    NEW_SESSION = st.sidebar.button("New Session")

    if generic_leader.storage:
        if st.sidebar.button("Delete All Session", key="delete_all_session_button"):
            for id in generic_leader.storage.get_all_session_ids():
                generic_leader.storage.delete_session(id)
            NEW_SESSION = True

    if NEW_SESSION:
        logger.info(">>> Creating new session...")
        KEYS = list(st.query_params.keys())
        for key in KEYS:
            del st.query_params[key]
        restart_agent()


if check_password():
    main()
