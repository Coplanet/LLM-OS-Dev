from io import BytesIO
from typing import Union

import cv2
import numpy as np
import streamlit as st
from phi.agent import Agent
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas

from ai.tools.stability import Stability
from app.utils import rerun, run_js
from db.session import get_db_context
from db.tables.user_config import UserBinaryData, UserNextOp


@st.dialog("Draw a mask over the image", width="large")
def render_mask_image(agent: Agent) -> Union[None, bool]:
    image: bytes = Stability.latest_user_image(
        agent, type=UserBinaryData.IMAGE, return_data=True
    )

    file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, 1)
    h, w = image_cv2.shape[:2]

    WIDTH_LIMIT = 700
    if w > WIDTH_LIMIT:
        h_, w_ = int(h * WIDTH_LIMIT / w), WIDTH_LIMIT
    else:
        h_, w_ = h, w

    thicker_cols = st.columns(2)
    with thicker_cols[0]:
        option = st.selectbox(
            "Select the thicker size:",
            ("Thin", "Medium", "Thick"),
            index=1,
        )

    stroke_width = 10

    if option == "Thin":
        stroke_width = 10
    elif option == "Medium":
        stroke_width = 25
    else:
        stroke_width = 50

    canvas = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_image=PILImage.open(BytesIO(image)).resize((h_, w_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode="freedraw",
        key="canvas",
    )

    stroke = canvas.image_data

    col1, col2 = st.columns(2)
    with col1:
        if stroke is not None and st.button("Edit Image"):
            mask = cv2.split(stroke)[3]
            mask = np.uint8(mask)
            mask = cv2.resize(mask, (w, h))
            img_byte_arr = BytesIO()
            mask = PILImage.fromarray(mask)
            mask.save(img_byte_arr, format="png")

            with get_db_context() as db:
                UserNextOp.delete_all_by_key(
                    db, agent.session_id, UserNextOp.GET_IMAGE_MASK, auto_commit=False
                )
                UserNextOp.save_op(
                    db,
                    agent.session_id,
                    UserNextOp.EDIT_IMAGE_USING_MASK,
                    auto_commit=False,
                )
                UserBinaryData.save_data(
                    db,
                    agent.session_id,
                    UserBinaryData.IMAGE_MASK,
                    UserBinaryData.DOWNSTREAM,
                    img_byte_arr.getvalue(),
                    "image/webp",
                    "webp",
                )

            st.success("Image's mask has been added successfully.")
            run_js(
                """setTimeout(function() {
                    window.parent.document
                        .querySelectorAll('button[aria-label="Close"]')
                        .forEach(button => button.click());
                }, 1000);"""
            )
            rerun()

    with col2:
        if st.button("Cancel"):
            with get_db_context() as db:
                UserNextOp.delete_all_by_key(
                    db, agent.session_id, UserNextOp.GET_IMAGE_MASK
                )
            st.warning("Image editing has been cancelled!")
            rerun()
