from textwrap import dedent
from typing import List, Optional

from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.storage.agent.postgres import PgAgentStorage
from phi.tools import Toolkit
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.file import FileTools
from phi.tools.resend_tools import ResendTools
from phi.tools.yfinance import YFinanceTools
from phi.vectordb.pgvector import PgVector2

from ai.agents import (
    arxiv,
    github,
    google_calender,
    patent_writer,
    python,
    reporter,
    wikipedia,
    youtube,
)
from ai.agents.base import CitextAgentTeam, agent_settings
from db.session import db_url
from db.settings import db_settings
from workspace.settings import citex_settings

from .base import CitexGPT4Leader


def get_leader(
    calculator: bool = True,
    ddg_search: bool = True,
    file_tools: bool = True,
    finance_tools: bool = True,
    resend_tools: bool = True,
    python_assistant: bool = True,
    youtube_assistant: bool = True,
    arxiv_assistant: bool = True,
    journal_assistant: bool = True,
    wikipedia_assistant: bool = True,
    github_assistant: bool = True,
    google_calender_assistant: bool = True,
    patent_writer_assistant: bool = True,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []

    team_members = CitextAgentTeam()

    if python_assistant:
        team_members.activate(python.agent)
    if youtube_assistant:
        team_members.activate(youtube.agent)
    if arxiv_assistant:
        team_members.activate(arxiv.agent)
    if journal_assistant:
        team_members.activate(reporter.agent)
    if wikipedia_assistant:
        team_members.activate(wikipedia.agent)
    if github_assistant:
        team_members.activate(github.agent)
    if google_calender_assistant:
        team_members.activate(google_calender.agent)
    if patent_writer_assistant:
        team_members.activate(patent_writer.agent)

    if calculator:
        tools.append(
            Calculator(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
            )
        )

    if ddg_search:
        tools.append(DuckDuckGo(fixed_max_results=3))

    if finance_tools:
        tools.append(
            YFinanceTools(
                stock_price=True,
                company_info=True,
                analyst_recommendations=True,
                company_news=True,
            )
        )

    if file_tools:
        tools.append(FileTools(base_dir=citex_settings.scratch_dir))
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, "
            "and `list_files` to list files in the working directory."
        )

    if resend_tools:
        tools.append(
            ResendTools(
                api_key=citex_settings.resend_api_key,
                from_email="onboarding@resend.dev",
            )
        )
        extra_instructions.append(
            "Use your resend tool to send an email upon request, make sure your email is html formatted."
            "Send the email to hasanpour.tech@gmail.com"
        )

    knowledge_base = CombinedKnowledgeBase(
        sources=[
            PDFKnowledgeBase(
                path=citex_settings.knowledgebase_dir, reader=PDFReader(chunk=True)
            )
        ],
        # Store assistant knowledge base in ai.sql_assistant_knowledge table
        vector_db=PgVector2(
            db_url=db_url,
            collection="llm_os_documents",
            embedder=OpenAIEmbedder(
                model=agent_settings.embedding_model, dimensions=1536
            ),
        ),
        # 5 references are added to the prompt
        num_documents=5,
    )

    # Load the knowledge base
    knowledge_base.load(recreate=False)

    return CitexGPT4Leader.build(
        team_members,
        name="leader",
        role="Lead the team to complete the task",
        introduction=dedent(
            """\
            Hi, I'm your LLM OS.
            I have access to a set of tools and AI Assistants to assist you.
            Lets get started!\
            """
        ),
        description=dedent(
            """\
            You are the most advanced AI system in the world called `Citex LLM OS`.
            You have access to a set of tools and a team of AI Assistants at your disposal.
            Your goal is to assist the user in the best way possible.\
            """
        ),
        instructions=[
            dedent(
                """\
                When the user sends a message, first **think** and determine if:
                    - You can answer by using a tool available to you
                    - You need to search the knowledge base
                    - You need to search the internet
                    - You need to delegate the task to a team member
                    - You need to ask a clarifying question\
                """
            ).strip(),
            (
                "IMPORTANT: If the user asks about a topic, first ALWAYS search your knowledge "
                "base using the `search_knowledge_base` tool."
            ),
            (
                "IMPORTANT: If you dont find relevant information in your knowledge base, use the "
                "`duckduckgo_search` tool to search the internet."
            ),
            (
                "IMPORTANT: If the user puts a youtube link or url, delegate the task "
                "to Youtube Assistant team member, fetch the full caption of the video"
            ),
            (
                "IMPORTANT: If the user puts a wikipedia link or url or asks about wikipedia,"
                " delegate the task to Wikipedia Assistant team member"
            ),
            (
                "If the user asks to summarize the conversation, use the `get_chat_history` "
                "tool with None as the argument."
            ),
            "If the users message is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        # Introduce knowledge base to the leader
        knowledge_base=knowledge_base,
        # This setting adds a tool to search the knowledge base for information
        search_knowledge=True,
        # This setting adds a tool to get chat history
        read_chat_history=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_messages=6,
        # Set addication context to the system's prompt
        additional_context=extra_instructions,
        # Inject some app related items
        run_id=run_id,
        user_id=user_id,
        session_id=session_id,
        storage=PgAgentStorage(
            table_name="agent_sessions", db_url=db_settings.get_db_url()
        ),
    )
