import re
import time
import uuid
from textwrap import dedent

import streamlit as st
from streamlit.components.v1 import html


def to_label(string: str):
    return re.sub(r"[_]{2,}", "_", string.strip().lower().replace(" ", "_"))


def run_js(js: str, one_time: bool = True):
    parent_key = "js_container_{}_{}".format(int(time.time()), uuid.uuid4())
    with st.container(key=parent_key):
        if one_time:
            js = js.replace(
                "{cleanup_code}",
                dedent(
                    """
                    window.parent.document.querySelectorAll('.st-key-{}').forEach(e => e.remove());
                    """.format(
                        parent_key
                    )
                ).strip(),
            )
        js = f"<script>{js}</script>"
        html(js, height=0, width=0)


def close_dialog(timeout: int = 1000):
    run_js(
        """setTimeout(function() {{
            window.parent.document
                .querySelectorAll('button[aria-label="Close"]')
                .forEach(button => button.click());
            {{cleanup_code}}
        }}, {});""".format(
            timeout
        )
    )


def next_run_toast(toast: str, icon: str = "check_circle"):
    st.session_state["next_run_toast"] = {"toast": toast, "icon": icon}


def run_next_run_toast():
    if "next_run_toast" in st.session_state:
        st.toast(
            st.session_state["next_run_toast"]["toast"],
            icon=st.session_state["next_run_toast"]["icon"],
        )
        del st.session_state["next_run_toast"]


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


def scroll_to_bottom():
    # Scroll down the page to bottom
    run_js(
        """
        setTimeout(() => {
            window.parent.document
                .querySelector('[data-testid="stAppIframeResizerAnchor"]')
                .scrollIntoView({
                    behavior: 'smooth'
                });
            {cleanup_code}
        }, 100);
        """
    )
