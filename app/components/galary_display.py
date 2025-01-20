from typing import Union

import streamlit as st
from phi.agent import Agent
from streamlit.components.v1 import html
from streamlit_image_select import image_select

from app.utils import rerun
from db.session import get_db_context
from db.tables.user_config import UserBinaryData
from helpers.utils import binary2text


@st.dialog("Display Gallery", width="large")
def render_galary_display(agent: Agent) -> Union[None, bool]:
    selected_image = st.session_state.get("selected_image", 0)
    st.subheader("Gallery, click on an image to start working on it")
    with get_db_context() as db:
        images = UserBinaryData.get_data(db, agent.session_id, UserBinaryData.IMAGE)

        IMAGE_COUNT = images.count()

        if IMAGE_COUNT > 0 and selected_image == 0:
            selected_image = images.first().id
            st.session_state["selected_image"] = selected_image
        images_ = []
        index2id = {}
        selected_image_index = 0
        for index in range(IMAGE_COUNT - 1, -1, -1):
            image = binary2text(
                images[index].data,
                images[index].mimetype or "image/webp",
            )
            images_.append(image)
            index2id[index] = images[index].id
            if images[index].id == selected_image:
                selected_image_index = len(images_) - 1

        if selected_image == 0:
            selected_image_index = 0

        selected_image = image_select(
            "Please select an image for further processing...",
            images_,
            return_value="index",
            index=selected_image_index,
            use_container_width=False,
        )
        my_js = f"""
        const interval = setInterval(() => {{
            const iframes = window.parent.document.querySelectorAll("iframe.stIFrame");

            for (let iframe of iframes) {{
                if (iframe.contentWindow.window === window) {{
                    iframe.style.display = "none";
                    break;
                }}
            }}

            const component_iframes = window.parent.document.querySelectorAll("iframe.stCustomComponentV1");
            for (let iframe of component_iframes) {{
                if (iframe && iframe.contentWindow && iframe.contentDocument) {{
                    const iframeDocument = iframe.contentDocument || iframe.contentWindow.document;

                    const imgs = iframeDocument.querySelectorAll(".image-box");

                    imgs.forEach((e) => e.classList.remove("selected"));
                    if (imgs[{selected_image_index}]) {{
                        imgs[{selected_image_index}].classList.add("selected");
                        clearInterval(interval);
                        break;
                    }}
                }}
            }}
        }}, 30);
        """

        # Wrapt the javascript as html code
        my_html = f"<script>{my_js}</script>"
        html(my_html, height=0, width=0)

    cols = st.columns(2)
    with cols[1]:
        if st.button(
            "Continue",
            icon=":material/check_circle:",
            type="primary",
        ):
            st.session_state["selected_image"] = index2id[
                IMAGE_COUNT - selected_image - 1
            ]
            rerun()

    with cols[0]:
        if st.button("Cancel", icon=":material/close:", type="secondary"):
            rerun()
