from typing import Union

import streamlit as st
from phi.agent import Agent
from st_clickable_images import clickable_images

from db.session import get_db_context
from db.tables.user_config import UserBinaryData
from helpers.utils import binary2text


@st.dialog("Display Gallery", width="large")
def render_galary_display(agent: Agent) -> Union[None, bool]:
    index_extracted = False
    selected_image = st.session_state.get("selected_image", 0)
    st.subheader("Gallery, click on an image to start working on it")
    with get_db_context() as db:
        images = UserBinaryData.get_data(db, agent.session_id, UserBinaryData.IMAGE)
        IMAGE_COUNT = images.count()
        image_list = []
        for index in range(IMAGE_COUNT - 1, -1, -1):
            image_list.append(
                binary2text(images[index].data, images[index].mimetype or "image/webp")
            )
            if not index_extracted and images[index].id == selected_image:
                selected_image = index
                index_extracted = True

        new_selected_image = clickable_images(
            image_list,
            titles=[f"Image #{(i + 1)}" for i in range(IMAGE_COUNT)],
            div_style={
                "display": "flex",
                "justify-content": "center",
                "flex-wrap": "wrap",
            },
            img_style={"margin": "5px", "height": "200px"},
        )
        if not isinstance(new_selected_image, int) or (
            isinstance(new_selected_image, int) and new_selected_image < 0
        ):
            display_selected_image = selected_image
        else:
            display_selected_image = IMAGE_COUNT - new_selected_image - 1
        if display_selected_image <= 0:
            display_selected_image = 0
        if display_selected_image >= IMAGE_COUNT:
            display_selected_image = IMAGE_COUNT - 1
        st.markdown("---")
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.markdown("## Selected image: ")
        with cols[1]:
            st.image(
                binary2text(
                    images[display_selected_image].data,
                    images[display_selected_image].mimetype or "image/webp",
                ),
                use_column_width=True,
            )

    cols = st.columns(2)
    with cols[1]:
        if st.button("Selected"):
            st.session_state["selected_image"] = images[display_selected_image].id
            st.rerun()

    with cols[0]:
        if st.button("Cancel"):
            st.rerun()
