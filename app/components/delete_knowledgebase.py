import streamlit as st
from phi.agent import Agent

from app.utils import rerun


@st.dialog("Delete Knowledgebase?", width="large")
def render_delete_knowledgebase(agent: Agent):
    st.subheader("Are you sure you want to delete the knowledgebase?")
    st.markdown(
        "This action is irreversible and will remove all your knowledge from the system. "
        "You will not be able to retrieve it later."
    )
    cols = st.columns(2)
    with cols[1]:
        if st.button("Delete", icon=":material/delete:"):
            agent.knowledge.vector_db.delete()
            st.toast("Knowledge base deleted")
            rerun()
    with cols[0]:
        if st.button("Cancel", icon=":material/close:", type="primary"):
            rerun()
