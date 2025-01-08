from io import BytesIO
from typing import Union

import cv2
import numpy as np
import streamlit as st
from phi.agent import Agent
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas

from db.session import get_db_context
from db.tables.user_config import UserBinaryData, UserNextOp


@st.dialog("Draw a mask over the image", width="large")
def render_mask_image(agent: Agent) -> Union[None, bool]:
    with get_db_context() as db:
        image: bytes = (
            UserBinaryData.get_data(
                db, agent.session_id, UserBinaryData.IMAGE, UserBinaryData.DOWNSTREAM
            )
            .first()
            .data
        )

    file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, 1)
    h, w = image_cv2.shape[:2]

    if w > 800:
        h_, w_ = int(h * 800 / w), 800
    else:
        h_, w_ = h, w

    stroke_width = st.slider("Stroke width: ", 1, 50, 25)
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
            mask.save(img_byte_arr, format="webp")

            with get_db_context() as db:
                UserNextOp.delete_all_by_key(
                    db, agent.session_id, UserNextOp.GET_IMAGE_MASK, auto_commit=False
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
            st.rerun()

    with col2:
        if st.button("Cancel"):
            stroke_width.empty()
            canvas.empty()
            st.warning("Image editing has been cancelled!")
            st.rerun()
