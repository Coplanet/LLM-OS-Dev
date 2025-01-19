import re
import time
import uuid

import streamlit as st
from streamlit.components.v1 import html


def to_label(string: str):
    return re.sub(r"[_]{2,}", "_", string.strip().lower().replace(" ", "_"))


def run_js(js: str):
    key = "{}.{}".format(time.time(), uuid.uuid4())
    with st.container(key=f"js_container_{key}"):
        js = f"<script>{js}</script>"
        html(js, height=0, width=0)


def rerun(clean_session: bool = False):
    if clean_session:
        keys2delete = []
        for key in st.session_state:
            keys2delete.append(key)

        for key in keys2delete:
            try:
                del st.session_state[key]
            except Exception:
                pass
    else:
        st.session_state["rerun"] = True

    st.rerun()
