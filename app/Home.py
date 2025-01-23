import hashlib
import os
import re
import tempfile
from datetime import datetime, timedelta
from os import getenv
from time import sleep, time
from typing import Dict, List, Optional
from urllib.parse import urlencode
from uuid import uuid4

import anthropic
import backoff
import google.generativeai as genai
import groq
import nest_asyncio
import openai
import sqlalchemy as sql
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
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
from phi.tools.streamlit.components import check_password
from phi.utils.log import logger as phi_logger
from phi.utils.log import logging
from streamlit_float import float_init

from ai.agents import base, settings
from ai.agents.base import Provider
from ai.agents.settings import agent_settings
from ai.agents.voice_transcriptor import voice2prompt
from ai.coordinators import generic as coordinator
from ai.document.reader.excel import ExcelReader
from ai.document.reader.general import GenericReader
from ai.document.reader.image import ImageReader
from ai.document.reader.pptx import PPTXReader
from app.auth import Auth, User
from app.components.available_agents import get_available_agents
from app.components.composio_integrations import composio_integrations
from app.components.configs import IMAGE_DIR
from app.components.delete_knowledgebase import render_delete_knowledgebase
from app.components.galary_display import render_galary_display
from app.components.mask_image import render_mask_image
from app.components.popup import show_popup
from app.components.sidebar import create_sidebar
from app.components.styles import render_styles
from app.models import AUDIO_SUPPORTED_MODELS
from app.utils import rerun, run_js, run_next_run_toast, scroll_to_bottom
from db.session import get_db_context
from db.settings import db_settings
from db.tables import UserBinaryData, UserConfig, UserIntegration, UserNextOp
from helpers.log import logger
from helpers.utils import binary2text, binary_text2data, text2binary
from workspace.settings import extra_settings

phi_logger.setLevel(logging.DEBUG)

auth = Auth()

nest_asyncio.apply()
st.set_page_config(page_title="CoPlanet AI", page_icon=f"{IMAGE_DIR}/favicon.png")

render_styles()

st.title("CoPlanet AI")
with st.container(key="subtitle_container"):
    st.markdown(
        """\
        LLM OS: Where Agents Meet Creativity\
        """,
        unsafe_allow_html=True,
    )


user: User = auth.get_user()


def backoff_handler(details):
    logger.error(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, groq.RateLimitError, anthropic.APIStatusError),
    max_time=60,
    max_tries=10,
    on_backoff=backoff_handler,
)
def run(
    generic_leader: Agent,
    question: str,
    uploaded_images: List[Image],
    uploaded_videos_: List[Image],
    audio_bytes: bytes,
    response_in_voice: bool,
    AUDIO_RESPONSE_SUPPORT: bool,
    audio_bytes_: bytes,
):

    response = ""
    resp_container = st.empty()
    start = time()

    if not AUDIO_RESPONSE_SUPPORT or not response_in_voice:
        try:
            # THIS IS A HACK TO SUPPORT IMAGES IN ANTHROPIC
            # Phidata doesn't support base64 images in Anthropic
            if generic_leader.model.provider == Provider.Anthropic.value:
                imgs = [text2binary(img) for img in uploaded_images]
                uploaded_images = imgs

            for delta in generic_leader.run(
                message=question,
                images=uploaded_images,
                videos=uploaded_videos_,
                audio=audio_bytes,
                stream=True,
            ):
                response += delta.content  # type: ignore
                response = re.sub(r"[\n\s]*!\[[^\]]+?\]\([^\)]+?\)", "", response)
                resp_container.markdown(response)

            # THIS IS A HACK TO SUPPORT IMAGES IN ANTHROPIC
            # Phidata doesn't support base64 images in Anthropic
            if generic_leader.model.provider == Provider.Anthropic.value:
                imgs = [text2binary(img) for img in uploaded_images]
                uploaded_images = imgs

        except Exception as e:
            logger.exception(e)
            st.exception(e)
    else:
        try:
            generic_leader.run(
                message="Answer the input audio.",
                images=uploaded_images,
                videos=uploaded_videos_,
                audio={
                    "data": binary_text2data(audio_bytes_),
                    "format": "wav",
                },
            )

        except Exception as e:
            logger.exception(e)
            st.exception(e)

    end = time()
    logger.debug(
        "Time to response from coordinator: {:.2f} seconds".format(end - start)
    )
    return response


def get_selected_assistant_config(user: User, label, package):
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
            "model_type": Provider.OpenAI.value,
            "model_id": "gpt-4o",
            "model_kwargs": {},
            "temperature": 0,
            "enabled": True,
            "max_tokens": agent_settings.default_max_completion_tokens,
            "tools": {k: True for k in available_tools_manifest.keys()},
        }

        with get_db_context() as db:
            config: UserConfig = UserConfig.get_models_config(
                db, user.session_id, auto_create=False
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
            user=user,
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

    run_next_run_toast()

    RERUN_SESSION = False
    if "rerun" in st.session_state:
        RERUN_SESSION = True
        del st.session_state["rerun"]

    USERNAME: str = user.username

    if getenv("RUNTIME_ENV") == "dev":
        USERNAME: str = "CoPlanet"

    if not USERNAME:
        # Improved message for password acceptance
        st.toast("Welcome aboard! Your password has been successfully verified.")
        USERNAME = user.get_username()
        if not USERNAME:
            return

    if not user.is_authenticated:
        user.username = USERNAME
        user.session_id = str(uuid4())
        user.to_auth_param(add_to_query_params=True)
        rerun()

    st.sidebar.info(f":label: User: {user.username}")

    st.sidebar.markdown("### Connect your apps to LLM OS")
    st.sidebar.button(
        "Integrate with Composio",
        on_click=lambda: composio_integrations(user),
        type="primary",
    )
    st.sidebar.markdown("---")

    if "callback" in st.query_params:
        source = st.query_params.get("source")
        if source == "composio":
            app = st.query_params.get("app")
            if app:
                hash = st.query_params.get("hash")
                if (
                    not hash
                    or hash
                    != hashlib.sha256(
                        f"{extra_settings.secret_key}:{user.username}:{user.session_id}:{app}:{source}".encode()
                    ).hexdigest()
                ):
                    st.toast("Invalid request!", icon=":material/error:")
                    rerun(True)
                    return
                status = st.query_params.get("status")
                connection_id = st.query_params.get("connectedAccountId")

                if str(status).lower() == "success" and connection_id:
                    if UserIntegration.add_integration(
                        user.user_id, app, connection_id, user.to_dict()
                    ):
                        st.toast("Connected to " + app + "!", icon=":material/check:")
                    else:
                        st.toast(
                            "Connection to " + app + " already exists!",
                            icon=":material/warning:",
                        )

                    for key in list(st.query_params.keys()):
                        del st.query_params[key]

                    user.to_auth_param(add_to_query_params=True)
                    rerun(clean_session=True)
                    return

                else:
                    st.toast(
                        "Failed to connect to " + app + "!", icon=":material/error:"
                    )
    # Initialize session state for popup control
    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

    # Get the Agent
    generic_leader: Agent

    AGENTS = get_available_agents(user)
    COORDINATOR_CONFIG = settings.AgentConfig.empty(user)
    AGENTS_CONFIG: Dict[str, settings.AgentConfig] = {}

    logger.debug(">>> Identified Session ID: %s", user)

    for agent, agent_config in AGENTS.items():
        config = get_selected_assistant_config(
            user,
            agent_config.get("label", agent),
            agent_config.get("package"),
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
        or st.session_state["generic_leader"].session_id != user.session_id
    ):
        logger.debug(">>> Creating leader agent with config: %s", COORDINATOR_CONFIG)
        coordinator.agent = generic_leader = coordinator.get_coordinator(
            team_config=AGENTS_CONFIG,
            config=COORDINATOR_CONFIG,
            session_id=user.session_id,
            user_id=user.user_id,
        )
        generic_leader.create_session()
        st.session_state["generic_leader"] = generic_leader
        if "CONFIG_CHANGED" in st.session_state:
            del st.session_state["CONFIG_CHANGED"]
    else:
        generic_leader = st.session_state["generic_leader"]

    for agent in [generic_leader] + generic_leader.team:
        agent: base.Agent

        config: settings.AgentConfig = AGENTS_CONFIG.get(
            agent.name
        ) or settings.AgentConfig.empty(user)

        st.session_state[f"{agent.label}_model_id"] = config.model_id or agent.model.id
        st.session_state[f"{agent.label}_model_type"] = (
            config.provider or agent.model_type
        )
        st.session_state[f"{agent.label}_temperature"] = config.temperature or getattr(
            agent.model, "temperature", 0
        )
        st.session_state[f"{agent.label}_max_tokens"] = config.max_tokens or getattr(
            agent.model,
            "max_tokens",
            agent_settings.default_max_completion_tokens,
        )

    # Create sidebar
    create_sidebar(user.session_id, {generic_leader.name: AGENTS[generic_leader.name]})

    # Sidebar checkboxes for selecting team members
    st.sidebar.markdown("### Select Team Members")
    create_sidebar(
        user.session_id,
        {k: v for k, v in AGENTS.items() if not v.get("is_leader", False)},
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
        show_popup(
            package.agent, user.session_id, selected_assistant, agent_config, package
        )
        st.session_state.show_popup = False
        st.session_state.selected_assistant = None

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # Load existing messages
    st.session_state["messages"] = generic_leader.memory.messages

    if not st.session_state["messages"]:
        if "q" in st.query_params:
            with st.chat_message("user"):
                st.markdown(st.query_params["q"])
                question = st.query_params["q"]

    audio_bytes = None
    AUDIO_ERROR = None
    response_in_voice = False
    AUDIO_RESPONSE_SUPPORT = False

    if False:
        with st.sidebar:
            with st.container(key="voice_input_container"):
                AUDIO_RESPONSE_SUPPORT = (
                    COORDINATOR_CONFIG.provider == Provider.OpenAI.value
                    and COORDINATOR_CONFIG.model_id
                    in AUDIO_SUPPORTED_MODELS[Provider.OpenAI.value]
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
                    "Response in Voice",
                    value=False,
                    disabled=not AUDIO_RESPONSE_SUPPORT,
                )
                if not AUDIO_RESPONSE_SUPPORT:
                    st.warning(
                        "You need to user following models in OpenAI to receive audio response: {}".format(
                            ", ".join(AUDIO_SUPPORTED_MODELS[Provider.OpenAI.value])
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
        audio_bytes = binary2text(audio_bytes, "audio/wav")
        # Here you can add logic to process the audio message
        st.session_state["messages"].append(
            {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
        )

    if (
        "uploaded_images" not in st.session_state
        or "hash2uploaded_images" not in st.session_state
    ):
        st.session_state["uploaded_images"] = []
        st.session_state["hash2uploaded_images"] = {}
        with get_db_context() as db:
            for d in UserBinaryData.get_data(
                db,
                user.session_id,
                UserBinaryData.IMAGE,
                UserBinaryData.DOWNSTREAM,
            ):
                st.session_state["hash2uploaded_images"][d.data_compressed_hashsum] = d
                st.session_state["uploaded_images"].append(d)
            st.session_state["uploaded_images"].reverse()

    st.session_state["rendered_images_hashes"] = set()
    hash2images = st.session_state.get("hash2uploaded_images", {})

    last_prompt = None
    last_user_message_index = None
    messages2remove = []

    previous_message_hash = None

    try:
        # Display existing chat messages
        for index, message in enumerate(generic_leader.memory.messages):
            if isinstance(message.content, str):
                message.content = re.sub(
                    r"[\n\s]*!\[[^\]]+?\]\([^\)]+?\)", "", message.content
                ).strip()

            if not message.content:
                continue

            message_role = message.role

            message_hash = hashlib.sha256(
                "{}/{}".format(message_role, message.content).encode()
            ).hexdigest()

            if previous_message_hash == message_hash:
                messages2remove.append(index)
                continue

            previous_message_hash = message_hash

            # Skip system and tool messages
            if message.role in ["system", "tool", "developer", "model"]:
                continue

            if message_role is not None:
                # Skip audio messages for now
                if message_role == "user" and isinstance(message.content, list):
                    if (
                        message.content[0].get("type") == "audio"
                        or message.content[0].get("type") == "tool_result"
                    ):
                        continue

                if message_role == "assistant" and message.content:
                    ct = message.content
                    if not isinstance(ct, list):
                        ct = [ct]

                    skip = False
                    for item in ct:
                        if isinstance(item, dict):
                            if "tool" in item.get("type"):
                                skip = True
                                break
                    if skip:
                        continue

                chat_message_container = st.chat_message(
                    message_role,
                    avatar="user" if message_role == "user" else "assistant",
                )
                if message_role == "user":
                    last_user_message_index = index
                with chat_message_container:
                    content = message.content
                    if isinstance(content, list):
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if item["type"] == "text":
                                last_prompt = item["text"]
                                st.write(last_prompt)
                            elif item["type"] == "image":
                                if message_role == "user":
                                    continue
                                if "source" in item:
                                    data_ = item["source"]["data"]
                                    if not data_.startswith("http"):
                                        hash = hashlib.sha256(
                                            text2binary(data_)
                                        ).hexdigest()
                                        if (
                                            hash
                                            in st.session_state[
                                                "rendered_images_hashes"
                                            ]
                                        ):
                                            continue
                                        data_ = hash2images.get(
                                            hash, item["image_url"]["url"]
                                        )
                                        if message_role == "user":
                                            st.session_state[
                                                "rendered_images_hashes"
                                            ].add(hash)
                                    st.image(
                                        (
                                            data_.data
                                            if isinstance(data_, UserBinaryData)
                                            else data_
                                        ),
                                        caption=item.get("image_caption"),
                                        use_column_width=True,
                                    )
                                else:
                                    continue
                            elif item["type"] == "image_url":
                                if message_role == "user":
                                    continue
                                data_ = item["image_url"]["url"]
                                if not data_.startswith("http"):
                                    hash = hashlib.sha256(
                                        text2binary(data_)
                                    ).hexdigest()
                                    if (
                                        hash
                                        in st.session_state["rendered_images_hashes"]
                                    ):
                                        continue
                                    data_ = hash2images.get(
                                        hash, item["image_url"]["url"]
                                    )
                                    if message_role == "user":
                                        st.session_state["rendered_images_hashes"].add(
                                            hash
                                        )
                                st.image(
                                    (
                                        data_.data
                                        if isinstance(data_, UserBinaryData)
                                        else data_
                                    ),
                                    caption=item.get("image_caption"),
                                    use_column_width=True,
                                )
                            # We disable audio rendering for now
                            elif item["type"] == "audio" and False:
                                st.audio(
                                    text2binary(item["audio"]),
                                    format="audio/wav",
                                )
                    else:
                        st.write(content)
    except Exception as e:
        st.exception(e)

    if messages2remove:
        messages2remove.reverse()
        for index in messages2remove:
            generic_leader.memory.messages.pop(index)
        generic_leader.write_to_storage()

    float_init()
    footer_container = st.container(key="footer_container")
    with footer_container:
        if prompt := st.chat_input(placeholder="How can CoPlanet LLM-OS assist you?"):
            st.session_state["messages"].append(Message(role="user", content=prompt))

    try:
        footer_container.float(
            "display:flex; align-items:center;justify-content:center; overflow:hidden visible;flex-direction:column; position:fixed;bottom:15px;"
        )

    except Exception as e:
        logger.error(f"Error floating footer container: {e}")

    if prompt:
        with st.chat_message("user", avatar="user"):
            st.write(prompt)
        scroll_to_bottom()

    if RERUN_SESSION or "page_loaded" not in st.session_state:
        st.session_state["page_loaded"] = True
        scroll_to_bottom()

    # If last message is from a user, generate a new response
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )

    with get_db_context() as db:
        mask_captured = False
        if (
            UserNextOp.get_op(db, user.session_id, UserNextOp.GET_IMAGE_MASK)
            and render_mask_image(generic_leader)
        ) or (
            mask_captured := UserNextOp.get_op(
                db, user.session_id, UserNextOp.EDIT_IMAGE_USING_MASK
            )
        ):
            last_message = st.session_state["messages"][last_user_message_index]
            if isinstance(last_message.content, list):
                for item in last_message.content:
                    if item.get("type") == "text":
                        last_message.content = item["text"]
                        break
            if mask_captured:
                with st.chat_message("assistant", avatar="assistant"):
                    st.write("Captured the mask, processing...")
                prefix = "**Mask captured, Proceed with:** "
                last_message.content = "{}{}".format(
                    prefix, last_message.content.replace(prefix, "")
                )

    if last_message and last_message.role == "user":
        question = last_message.content
        if isinstance(question, list):
            if question[0].get("type") == "audio":
                current_audio_uploaded = bool(audio_bytes)
                audio_bytes = binary_text2data(question[0]["audio"])
                if current_audio_uploaded:
                    generic_leader.memory.add_message(
                        Message(role="user", content=last_message.content)
                    )
                    generic_leader.write_to_storage()

        with st.chat_message("assistant"):
            uploaded_videos_ = []
            uploaded_videos = st.session_state.get("uploaded_videos", [])
            if uploaded_videos:
                if COORDINATOR_CONFIG.provider != Provider.Google.value:
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

            uploaded_images = st.session_state["uploaded_images"]

            with st.spinner("Thinking..."):
                selected_image = st.session_state.get("selected_image", None)

                image_data = None
                image_type = "image/webp"

                if not selected_image and uploaded_images:
                    img: UserBinaryData = uploaded_images[len(uploaded_images) - 1]
                    if img:
                        image_data = img.data_compressed
                        image_type = img.mimetype or "image/webp"

                if not image_data and not selected_image and not uploaded_images:
                    with get_db_context() as db:
                        image = UserBinaryData.get_data(
                            db,
                            generic_leader.session_id,
                            UserBinaryData.IMAGE,
                        ).first()
                        if image:
                            image_data = image.data_compressed
                            selected_image = image.id
                            image_type = image.mimetype or "image/webp"

                response = run(
                    generic_leader,
                    question,
                    [binary2text(image_data, image_type)] if image_data else [],
                    uploaded_videos_,
                    audio_bytes,
                    response_in_voice,
                    AUDIO_RESPONSE_SUPPORT,
                    audio_bytes_,
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

            requires_update = False
            # Get the images
            image_outputs: Optional[List[Image]] = generic_leader.get_images()

            with get_db_context() as db:
                if UserNextOp.get_op(db, user.session_id, UserNextOp.GET_IMAGE_MASK):
                    if render_mask_image(generic_leader):
                        run(
                            generic_leader,
                            question,
                            uploaded_images,
                            uploaded_videos_,
                            audio_bytes,
                            response_in_voice,
                            AUDIO_RESPONSE_SUPPORT,
                            audio_bytes_,
                        )

            # Render the images
            if image_outputs:
                image_outputs_ = {}
                for index in range(len(image_outputs)):
                    img = image_outputs[index]
                    if isinstance(img, dict):
                        image_outputs[index] = Image.model_validate(img)
                    image_outputs_[img.id] = img
                image_outputs = list(image_outputs_.values())
                logger.debug("Rendering '{}' images...".format(len(image_outputs)))
                contents = []
                contents.append({"type": "text", "text": response})

                for img in image_outputs:
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
                requires_update = True

            else:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

            if requires_update:
                generic_leader.write_to_storage()

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
            st.session_state["file_uploader_key"] = "uploader.{}.{}".format(
                time(), uuid4()
            )

        if "uploaded_images" not in st.session_state or not isinstance(
            st.session_state["uploaded_images"], list
        ):
            st.session_state["uploaded_images"] = []

        def file_uploaded_on_change():
            st.toast("Uploading...", icon=":material/upload:")

        uploaded_files_ = st.sidebar.file_uploader(
            "Add a Document (video & image files, .pdf, .csv, .pptx, .txt, .md, .docx, .json, .xlsx, .xls and etc)",
            key=st.session_state["file_uploader_key"],
            accept_multiple_files=True,
            on_change=file_uploaded_on_change,
        )
        if uploaded_files_:
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
                                    uploaded_images.append(text2binary(image.content))
                        else:
                            st.sidebar.error(
                                "Could not read document: {}".format(document_name)
                            )
                        st.session_state[f"{document_name}_uploaded"] = True
                        st.toast(
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

            if "hash2uploaded_images" not in st.session_state:
                st.session_state["hash2uploaded_images"] = {}

            with get_db_context() as db:
                images = UserBinaryData.save_bulk(
                    db,
                    user.session_id,
                    UserBinaryData.IMAGE,
                    UserBinaryData.DOWNSTREAM,
                    uploaded_images,
                )
                for image in images:
                    st.session_state["hash2uploaded_images"][
                        image.data_compressed_hashsum
                    ] = image

                if images:
                    st.session_state["uploaded_images"].extend(images)
                    st.session_state["selected_image"] = images[-1].id

    with footer_container:
        IMAGE_UPLOAED = bool(
            UserBinaryData.get_data(db, user.session_id, UserBinaryData.IMAGE).count()
            > 0
        )
        KNOWLEDGE_BASE_CREATED = bool(generic_leader.knowledge.vector_db)
        # for upload
        columns = [0.05]
        COL_INDEX = 0
        if IMAGE_UPLOAED:
            columns.append(0.05)
        if KNOWLEDGE_BASE_CREATED:
            columns.append(0.05)
        # for new session
        columns.append(0.05)
        # for in the middle
        columns.insert(-1, 1 - sum(columns))
        cols = st.columns(columns)
        with cols[COL_INDEX]:
            COL_INDEX += 1
            run_js(
                """document.addEventListener('DOMContentLoaded', function () {
                    const interval = setInterval(() => {
                        const button = window.parent.document.querySelector('.st-key-cloud_upload button');
                        if (button) {
                            button.addEventListener('click', function (e) {
                                e.preventDefault();
                                const fileUploader = window.parent.document.querySelector('input[type="file"]');
                                if (fileUploader) {
                                    fileUploader.click();
                                }
                            });
                            clearInterval(interval);
                            {cleanup_code}
                        }
                    }, 100);
                })"""
            )
            st.button(":material/cloud_upload:", key="cloud_upload")

        with get_db_context() as db:
            if IMAGE_UPLOAED:
                with cols[COL_INDEX]:
                    COL_INDEX += 1
                    if st.button(
                        ":material/gallery_thumbnail:", key="gallery_thumbnail"
                    ):
                        render_galary_display(generic_leader)

        if KNOWLEDGE_BASE_CREATED:
            with cols[COL_INDEX]:
                COL_INDEX += 1
                if st.button(":material/delete:", key="delete_knowledge_base"):
                    render_delete_knowledgebase(generic_leader)

        with cols[-1]:
            NEW_SESSION = st.button(":material/add_box:", key="add_box")

    EMPTY_SESSIONS = True

    st.sidebar.markdown("---")

    if generic_leader.storage:
        sessions = generic_leader.storage.get_all_sessions(user_id=user.user_id)
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
                topics = []
                for topic in session.memory["summary"]["topics"]:
                    if len(topics) >= 3:
                        break
                    # purge html from topic and only add if not empty do it with beautifulsoup
                    topic = BeautifulSoup(topic, "html.parser").get_text().strip()
                    if topic:
                        topics.append(topic)

                if not topics:
                    topics.append("No topics found")

                session_info = {
                    "id": session.session_id,
                    "topics": ", ".join(topics),
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
        for period in ["Today", "Yesterday", "Previous 7 Days"]:
            sessions = session_options[period]
            if not sessions:
                continue

            EMPTY_SESSIONS = False
            st.sidebar.markdown(f"### {period}")
            for session in sessions:
                topics = session["topics"]
                link = "?{}".format(
                    urlencode(
                        User(user.username, session_id=session["id"]).to_auth_param()
                    )
                )
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
                options.index(user.session_id) if user.session_id in options else 0
            )
            if options:

                def on_older_session_change():
                    user.session_id = st.session_state["older_sessions_selectbox"]
                    user.to_auth_param(add_to_query_params=True)
                    rerun()

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
        CLEAN_SESSION = True

        if generic_leader.storage:
            if st.sidebar.button("Delete All Session", key="delete_all_session_button"):
                ids = generic_leader.storage.get_all_session_ids(user_id=user.user_id)
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
            user.session_id = str(uuid4())
            user.to_auth_param(add_to_query_params=True)
            rerun(clean_session=CLEAN_SESSION)


if (
    os.getenv("RUNTIME_ENV") != "prd"
    or user.is_authenticated
    or (True or check_password())
):
    main()
