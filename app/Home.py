import base64
import hashlib
import os
import re
import tempfile
from datetime import datetime, timedelta
from io import BytesIO
from os import getenv
from time import sleep, time
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode
from uuid import uuid4

import google.generativeai as genai
import nest_asyncio
import sqlalchemy as sql
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
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
from phi.storage.agent.postgres import PgAgentStorage
from phi.tools.streamlit.components import check_password, get_username_sidebar
from phi.utils.log import logger as phi_logger
from phi.utils.log import logging
from PIL import Image as PILImage

from ai.agents import base, settings
from ai.agents.settings import agent_settings
from ai.agents.voice_transcriptor import voice2prompt
from ai.coordinators import generic as coordinator
from ai.document.reader.excel import ExcelReader
from ai.document.reader.general import GenericReader
from ai.document.reader.image import ImageReader
from ai.document.reader.pptx import PPTXReader
from app.components.available_agents import AGENTS
from app.components.popup import AUDIO_SUPPORTED_MODELS, show_popup
from app.components.sidebar import create_sidebar
from db.session import get_db_context
from db.settings import db_settings
from db.tables import UserConfig
from helpers.log import logger
from helpers.utils import audio2text, audio_text2data, text2audio

phi_logger.setLevel(logging.DEBUG)

STATIC_DIR = "app/static"
IMAGE_DIR = f"{STATIC_DIR}/images"
CSS_DIR = f"{STATIC_DIR}/css"

SESSION_KEY = "sid"

nest_asyncio.apply()
st.set_page_config(page_title="CoPlanet AI", page_icon=f"{IMAGE_DIR}/favicon.png")

# load css
with open(f"{CSS_DIR}/main.css", "r") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


for theme in ["dark", "light"]:
    if os.path.exists(f"{CSS_DIR}/main-{theme}.css"):
        with open(f"{CSS_DIR}/main-{theme}.css", "r") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

components.html(
    """
<script>
document.addEventListener("DOMContentLoaded", function() {
    setInterval(() => {
        const appElement = window.parent.document.getElementsByClassName("stApp")[0];
        const currentTheme = window.getComputedStyle(appElement).getPropertyValue("color-scheme");

        // Remove existing theme classes
        appElement.classList.remove('dark', 'light');

        // Add the current theme class
        if (currentTheme === 'dark') {
            appElement.classList.add('dark');
        } else if (currentTheme === 'light') {
            appElement.classList.add('light');
        }
    }, 300);
});
</script>
""",
    height=0,
    width=0,
)

st.title("CoPlanet AI")
st.markdown(
    f"""\
    ##### <img src="{IMAGE_DIR}/coplanet.png" alt="Logo" style="display: none; width: 30px; margin-right: 10px;"> \
    CoPlanet LLM OS\
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.0/css/all.min.css">""",
    unsafe_allow_html=True,
)

# JavaScript to detect theme and attach color-scheme class to stApp
# st_javascript("""
# (function() {
#     const appElement = window.parent.document.getElementsByClassName("stApp")[0];
#     const currentTheme = window.getComputedStyle(appElement).getPropertyValue("color-scheme").trim();

#     // Remove existing theme classes
#     appElement.classList.remove('dark', 'light');

#     // Add the current theme class
#     if (currentTheme === 'dark') {
#         appElement.classList.add('dark');
#     } else if (currentTheme === 'light') {
#         appElement.classList.add('light');
#     }
# })();
# """, key="theme-detection-script")

# ... existing code ...


def restart_agent():
    logger.debug(">>> Restarting Agent")
    st.session_state["generic_leader"] = None
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
            "model_kwargs": {},
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
                    if tool.get("default_status", "enabled").lower() == "enabled":
                        tools[tool.get("name")] = tool

        return settings.AgentConfig(
            provider=default_configs["model_type"],
            model_id=default_configs["model_id"],
            model_kwargs=default_configs.get("model_kwargs", {}),
            temperature=default_configs["temperature"],
            enabled=default_configs["enabled"],
            max_tokens=default_configs["max_tokens"],
            tools=tools,
        )
    except Exception as e:
        logger.error("Error reading get_selected_assistant_config(): %s", e)
        return None


def main() -> None:
    if os.getenv("TESTING_ENV", False):
        return

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

    SID = (
        st.query_params[SESSION_KEY]
        if (SESSION_KEY in st.query_params and st.query_params[SESSION_KEY])
        else None
    )

    NEW_SESSION = not bool(SID)
    # Create Agent session (i.e. log to database) and save session_id in session state
    try:
        if NEW_SESSION:
            st.query_params[SESSION_KEY] = str(uuid4())
            st.rerun()
            return

    except Exception as e:
        logger.error(e)
        st.warning("Could not create Agent session, is the database running?")
        return

    # Initialize session state for popup control
    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

    # Get the Agent
    generic_leader: Agent

    COORDINATOR_CONFIG = settings.AgentConfig.empty()
    AGENTS_CONFIG: Dict[str, settings.AgentConfig] = {}

    if SID:
        logger.debug(">>> Identified Session ID: %s", SID)

        for agent, agent_config in AGENTS.items():
            config = get_selected_assistant_config(
                SID, agent_config.get("label", agent), agent_config.get("package")
            )
            if agent_config.get("is_leader", False):
                COORDINATOR_CONFIG = config
            else:
                AGENTS_CONFIG[agent] = config

        logger.debug("Agents Config: ")
        for agent, config in AGENTS_CONFIG.items():
            logger.debug(f"'{agent}' configed to: {config}")

        if (
            "CONFIG_CHANGED" in st.session_state
            or "generic_leader" not in st.session_state
            or st.session_state["generic_leader"] is None
            or st.session_state["generic_leader"].session_id != SID
        ):
            logger.debug(
                ">>> Creating leader agent with config: %s", COORDINATOR_CONFIG
            )
            coordinator.agent = generic_leader = coordinator.get_coordinator(
                team_config=AGENTS_CONFIG,
                config=COORDINATOR_CONFIG,
                session_id=SID,
                user_id=hashlib.md5(username.encode()).hexdigest(),
            )
            generic_leader.create_session()
            st.session_state["generic_leader"] = generic_leader
            if "CONFIG_CHANGED" in st.session_state:
                del st.session_state["CONFIG_CHANGED"]
        else:
            generic_leader = st.session_state["generic_leader"]

        for agent in [generic_leader] + generic_leader.team:
            agent: base.Agent

            config: settings.AgentConfig = (
                AGENTS_CONFIG.get(agent.name) or settings.AgentConfig.empty()
            )

            st.session_state[f"{agent.label}_model_id"] = (
                config.model_id or agent.model.id
            )
            st.session_state[f"{agent.label}_model_type"] = (
                config.provider or agent.model_type
            )
            st.session_state[f"{agent.label}_temperature"] = (
                config.temperature or getattr(agent.model, "temperature", 0)
            )
            st.session_state[f"{agent.label}_max_tokens"] = (
                config.max_tokens
                or getattr(
                    agent.model,
                    "max_tokens",
                    agent_settings.default_max_completion_tokens,
                )
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
                else COORDINATOR_CONFIG
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

    AUDIO_ERROR = None
    with st.sidebar:
        with st.container(key="voice_input_container"):
            AUDIO_RESPONSE_SUPPORT = (
                COORDINATOR_CONFIG.provider == "OpenAI"
                and COORDINATOR_CONFIG.model_id in AUDIO_SUPPORTED_MODELS["OpenAI"]
            )
            # define sample rate
            AUDIO_SAMPLE_RATE = 44_100
            # Add an audio recorder for voice messages
            if audio_bytes := audio_recorder(
                text="Voice Input",
                icon_size="1x",
                pause_threshold=5,
                sample_rate=AUDIO_SAMPLE_RATE,
                key="voice_input_recorder"
                + ("" if AUDIO_RESPONSE_SUPPORT else "_disabled"),
            ):
                # expecting the output should be byte
                if not isinstance(audio_bytes, bytes):
                    raise Exception("Recorded audio is not an instance of bytes")
                # reject audio with less than 2 seconds
                if len(audio_bytes) < 2 * 4 * AUDIO_SAMPLE_RATE:
                    AUDIO_ERROR = st.error(
                        "Recording cannot be less than 2 seconds!", icon="⚠"
                    )
                    audio_bytes = None

                if audio_bytes:
                    if AUDIO_ERROR:
                        AUDIO_ERROR.empty()
                        AUDIO_ERROR = None
                    logger.debug(
                        "Audio recorded: {:,} seconds".format(
                            len(audio_bytes) / (AUDIO_SAMPLE_RATE * 4)
                        )
                    )
            response_in_voice = st.checkbox(
                "Response in Voice", value=False, disabled=not AUDIO_RESPONSE_SUPPORT
            )
            if not AUDIO_RESPONSE_SUPPORT:
                st.warning(
                    "You need to user following models in OpenAI to receive audio response: {}".format(
                        ", ".join(AUDIO_SUPPORTED_MODELS["OpenAI"])
                    )
                )
            if response_in_voice:
                generic_leader.storage = None
            else:
                generic_leader.storage = PgAgentStorage(
                    table_name="agent_sessions", db_url=db_settings.get_db_url()
                )

    # Process the text or audio input
    if audio_bytes:
        audio_bytes = audio2text(audio_bytes)
        # Here you can add logic to process the audio message
        st.session_state["messages"].append(
            {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
        )

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
            # Skip audio messages for now
            if message_role == "user" and isinstance(message.get("content"), list):
                if message.get("content")[0].get("type") == "audio":
                    continue
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
                        # We disable audio rendering for now
                        elif item["type"] == "audio" and False:
                            st.audio(
                                text2audio(item["audio"]),
                                format="audio/wav",
                            )
                else:
                    st.write(content)

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]

    if last_message.get("role") == "user":
        question = last_message["content"]
        if isinstance(question, list):
            if question[0].get("type") == "audio":
                current_audio_uploaded = bool(audio_bytes)
                audio_bytes = audio_text2data(question[0]["audio"])
                if current_audio_uploaded:
                    generic_leader.memory.add_message(
                        Message(role="user", content=last_message["content"])
                    )
                    generic_leader.write_to_storage()

        with st.chat_message("assistant"):
            uploaded_videos_ = []
            uploaded_videos = st.session_state.get("uploaded_videos", [])
            if uploaded_videos:
                if COORDINATOR_CONFIG.provider != "Google":
                    st.error("Videos are only supported for Google provider")
                else:
                    with st.spinner("Uploading videos..."):
                        if "genai_uploaded_videos" not in st.session_state:
                            st.session_state["genai_uploaded_videos"] = {}

                        for video in uploaded_videos:
                            # if already uploaded?
                            if (
                                video.file_id
                                in st.session_state["genai_uploaded_videos"]
                            ):
                                uploaded_videos_.append(
                                    st.session_state["genai_uploaded_videos"][
                                        video.file_id
                                    ]
                                )
                                continue
                            # store audio bytes in a temp file
                            with tempfile.NamedTemporaryFile(
                                suffix="." + video.name.split(".")[-1]
                            ) as temp_file:
                                temp_file.write(video.read())
                                temp_file_path = temp_file.name
                                while True:
                                    gfile = genai.upload_file(
                                        temp_file_path, mime_type=video.type
                                    )
                                    if gfile and gfile.state.name != "FAILED":
                                        break
                                uploaded_videos_.append(gfile)

                            st.session_state["genai_uploaded_videos"][
                                video.file_id
                            ] = gfile

                        all_uploaded = False
                        while not all_uploaded:
                            all_uploaded = True
                            for gfile in uploaded_videos_:
                                if (
                                    genai.get_file(gfile.name).state.name
                                    == "PROCESSING"
                                ):
                                    all_uploaded = False
                                    sleep(1)
                                    break

            voice_transcribe: bool = False
            is_prompt: bool = False
            prompt: str = ""
            audio_bytes_ = audio_bytes

            if audio_bytes:
                with st.spinner("Listening..."):
                    start = time()
                    is_prompt, prompt, transcription = voice2prompt(audio_bytes)
                    end = time()
                    logger.debug(
                        "Time to voice2prompt: {:.2f} seconds".format(end - start)
                    )

                audio_bytes = None

                if is_prompt:
                    question = prompt

                else:
                    voice_transcribe = True
                    question = (
                        "Read the following text and respond to it: " + transcription
                    )
                question += "\n\n**RESPONSE WITH AUDIO.**"

            with st.spinner("Thinking..."):
                response = ""
                resp_container = st.empty()
                start = time()

                if not AUDIO_RESPONSE_SUPPORT or not response_in_voice:
                    for delta in generic_leader.run(
                        message=question,
                        images=uploaded_images,
                        videos=uploaded_videos_,
                        audio=audio_bytes,
                        stream=True,
                    ):
                        response += delta.content  # type: ignore
                        response = re.sub(
                            r"[\n\s]*!\[[^\]]+?\]\([^\)]+?\)", "", response
                        )
                        resp_container.markdown(response)
                else:
                    generic_leader.run(
                        message="Answer the input audio.",
                        images=uploaded_images,
                        videos=uploaded_videos_,
                        audio={"data": audio_text2data(audio_bytes_), "format": "wav"},
                    )

                end = time()
                logger.debug(
                    "Time to response from coordinator: {:.2f} seconds".format(
                        end - start
                    )
                )

            if AUDIO_RESPONSE_SUPPORT:
                if (
                    generic_leader.run_response.response_audio is not None
                    and "data" in generic_leader.run_response.response_audio
                ):
                    with st.container(key="response_audio"):
                        # flake8: noqa: E501
                        st.markdown(
                            f"""
                            <audio controls autoplay="true" style="display: none" id="audio-{generic_leader.run_response.response_audio["id"]}">
                                <source src="data:audio/wav;base64,{generic_leader.run_response.response_audio["data"]}" type="audio/wav">
                            </audio>
                            <script>
                                setTimeout(function() {{
                                    document.getElementById("audio-{generic_leader.run_response.response_audio["id"]}").play()
                                }}, 300)
                            </script>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("No audio response!")

            if not voice_transcribe and is_prompt and prompt:
                # remove the last role="user" message because since it's the generated prompt
                # we have already have the input voice in message history and we don't need to
                # store it the transcripted voice in the memory
                for i in range(len(generic_leader.memory.messages) - 1, -1, -1):
                    if generic_leader.memory.messages[i].role == "user":
                        del generic_leader.memory.messages[i]
                        break
                generic_leader.write_to_storage()

            if audio_bytes:
                audio_bytes = None

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
                image_outputs_ = {}
                for img in image_outputs:
                    image_outputs_[img.id] = img
                image_outputs = list(image_outputs_.values())
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
            "Add a Document (video & image files, .pdf, .csv, .pptx, .txt, .md, .docx, .json, .xlsx, .xls and etc)",
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
            uploaded_videos = []
            for uploaded_file in uploaded_files_:
                document_name = uploaded_file.name
                if f"{document_name}_uploaded" not in st.session_state:
                    file_type = uploaded_file.name.split(".")[-1].lower()

                    reader: Reader

                    if file_type in ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"]:
                        if "uploaded_videos" not in st.session_state:
                            st.session_state["uploaded_videos"] = []
                        st.session_state["uploaded_videos"].append(uploaded_file)
                        continue
                    elif file_type == "pdf":
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

    EMPTY_SESSIONS = True

    if generic_leader.storage:
        sessions = generic_leader.storage.get_all_sessions()
        session_options = {
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Older": [],
        }
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        for session in sessions:
            if "summary" in session.memory and "topics" in session.memory["summary"]:
                # Convert Unix timestamp to datetime
                session_date = datetime.fromtimestamp(session.created_at).date()
                session_info = {
                    "id": session.session_id,
                    "topics": ", ".join(session.memory["summary"]["topics"][:3]),
                }
                if session_date == today:
                    session_options["Today"].append(session_info)
                elif session_date == yesterday:
                    session_options["Yesterday"].append(session_info)
                elif session_date >= today - timedelta(days=7):
                    session_options["Previous 7 Days"].append(session_info)
                else:
                    session_options["Older"].append(session_info)

        # Display sessions
        rendered_hr = False
        for period in ["Today", "Yesterday", "Previous 7 Days"]:
            sessions = session_options[period]
            if not sessions:
                continue
            if not rendered_hr:
                st.sidebar.markdown("---")
                rendered_hr = True

            EMPTY_SESSIONS = False
            st.sidebar.markdown(f"### {period}")
            for session in sessions:
                session_id = session["id"]
                topics = session["topics"]
                query_string = urlencode({SESSION_KEY: session_id})
                link = f"?{query_string}"
                st.sidebar.markdown(
                    f"""
                    <div class="sidebar-session-link">
                        <a href="{link}" target="_self">
                            {topics}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if session_options["Older"]:
            sessions = session_options["Older"]
            options = [None] + [session["id"] for session in sessions]
            display = ["Pick an older session"] + [
                session["topics"] for session in sessions
            ]
            selectbox_index = (
                options.index(st.query_params[SESSION_KEY])
                if st.query_params[SESSION_KEY] in options
                else 0
            )
            if options:

                def on_older_session_change():
                    st.query_params[SESSION_KEY] = st.session_state[
                        "older_sessions_selectbox"
                    ]
                    st.rerun()

                st.sidebar.selectbox(
                    "### Select a session",
                    options=options,
                    format_func=lambda idx: display[options.index(idx)],
                    index=selectbox_index,
                    key="older_sessions_selectbox",
                    placeholder="Select an older session",
                    on_change=on_older_session_change,
                )

    if not EMPTY_SESSIONS:
        st.sidebar.markdown("---")
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
