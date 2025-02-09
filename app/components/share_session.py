import hashlib
from time import time
from typing import Optional

import streamlit as st

from app.auth import User
from app.utils import rerun
from helpers.log import logger
from workspace.settings import extra_settings


def generate_hash(user: User, now: int):
    return hashlib.sha256(
        "{}-{}-{}|{}".format(
            user.username, extra_settings.secret_key, user.session_id, now
        ).encode()
    ).hexdigest()


@st.dialog("Share This Session", width="large")
def share_session(user: User):
    st.subheader("Share this session to any body with the following link")
    st.markdown(
        "This link won't expire. Anyone with the link will be able to view the session."
        "Also this link **will not** expose your other topics and conversations."
    )

    now = int(time())
    hash = generate_hash(user, now)

    link = f"{extra_settings.app_url}?share=true&u={user.username},{now},{user.session_id}&h={hash}"

    st.code(link, language="url")

    st.markdown("---")
    if st.button("Close", icon=":material/close:", type="primary"):
        rerun()


def validate_share_session() -> Optional[User]:
    try:
        if "share" not in st.query_params:
            return None

        hash = st.query_params.get("h", "")
        username, timestamp, session_id = st.query_params.get("u", "").split(",")

        if (
            not all([username, session_id, hash, timestamp])
            or generate_hash(User(username, session_id), int(timestamp)) != hash
        ):
            return None

        return User(username, session_id)

    except Exception as e:
        logger.error(f"Error validating share session: {e}")
        return None
