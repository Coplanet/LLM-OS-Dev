import os

import streamlit as st
import streamlit.components.v1 as components

from app.components.configs import CSS_DIR


def render_styles():
    with st.container(key="css_and_theme_container"):
        st.markdown(
            """<link rel="stylesheet" href="{}">""".format(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.0/css/all.min.css"
            ),
            unsafe_allow_html=True,
        )
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
