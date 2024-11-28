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
from phi.utils.log import logger
from phi.vectordb.pgvector import PgVector2

from ai.agents import (
    arxiv,
    base,
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

from ..agents.settings import GPTModels
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
    model_id: Optional[str] = GPTModels.GPT4.value,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []

    team_members = CitextAgentTeam()

    def enable_agent_if(flag: bool, agent: base.CitexAgent):
        if flag:
            logger.info("Activating %s", agent.name)
            team_members.activate(agent)
        else:
            logger.info("DEACTICATING %s", agent.name)

    enable_agent_if(python_assistant, python.agent)
    enable_agent_if(youtube_assistant, youtube.agent)
    enable_agent_if(arxiv_assistant, arxiv.agent)
    enable_agent_if(journal_assistant, reporter.agent)
    enable_agent_if(wikipedia_assistant, wikipedia.agent)
    enable_agent_if(github_assistant, github.agent)
    enable_agent_if(google_calender_assistant, google_calender.agent)
    enable_agent_if(patent_writer_assistant, patent_writer.agent)

    if not calculator:
        logger.info("Removing Calculator tool with full functionality.")
    else:
        logger.info("Adding Calculator tool with full functionality.")
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
        extra_instructions.append(
            dedent(
                """\
                Use the Calculator tool for precise and complex mathematical operations, including addition,
                subtraction, multiplication, division, exponentiation, factorials, checking if a number is prime,
                and calculating square roots. This tool is ideal for mathematical queries, computations,
                or when the user needs help solving equations or understanding numeric concepts.\
                """
            ).strip()
        )

    if not ddg_search:
        logger.info("Removing DuckDuckGo search tool.")
    else:
        logger.info("Adding DuckDuckGo search tool with fixed max results: 3.")
        tools.append(DuckDuckGo(fixed_max_results=3))
        extra_instructions.append(
            dedent(
                """\
                Leverage the DuckDuckGo Search tool for quick internet searches, such as finding \
                    up-to-date information,
                verifying facts, or answering questions beyond the scope of the knowledge base.
                Use this tool when a direct query requires additional context or when you need to retrieve concise
                and relevant information (limited to 3 results per search).\
                """
            ).strip()
        )

    if not finance_tools:
        logger.info(
            "Removing YFinance Tools for stock price, company info, analyst recommendations, and company news."
        )
    else:
        logger.info(
            "Adding YFinance Tools for stock price, company info, analyst recommendations, and company news."
        )
        tools.append(
            YFinanceTools(
                stock_price=True,
                company_info=True,
                analyst_recommendations=True,
                company_news=True,
            )
        )
        extra_instructions.append(
            dedent(
                """\
                Utilize YFinance tools for financial and stock-related queries. This includes:

                - Stock Price: Retrieve real-time or historical stock prices.
                - Company Info: Fetch detailed company profiles or background.
                - Analyst Recommendations: Provide insights into stock recommendations from financial analysts.
                - Company News: Share recent news articles or updates related to the company.

                These tools are ideal for answering finance-related questions or assisting with investment decisions.\
                """
            ).strip()
        )

    if not file_tools:
        logger.info(
            "Removing File Tools with base directory: %s", citex_settings.scratch_dir
        )
    else:
        logger.info(
            "Adding File Tools with base directory: %s", citex_settings.scratch_dir
        )
        tools.append(FileTools(base_dir=citex_settings.scratch_dir))
        extra_instructions.append(
            """\
            Use the File Tools for managing files in the working directory. Specific use cases include:

            - Read Files: Open and read content from files when the user uploads or references one.
            - Save Files: Store data, results, or responses in a file upon request.
            - List Files: Display the contents of the working directory for easy navigation.

            This tool is helpful for file manipulation tasks, processing user-uploaded data, or \
            saving generated content for future use.\
            """
        )

    if not resend_tools:
        logger.info(
            "Removing Resend Tools with API key and from_email: %s",
            "onboarding@resend.dev",
        )
    else:
        logger.info(
            "Adding Resend Tools with API key and from_email: %s",
            "onboarding@resend.dev",
        )
        tools.append(
            ResendTools(
                api_key=citex_settings.resend_api_key,
                from_email="onboarding@resend.dev",
            )
        )
        extra_instructions.append(
            """\
            Employ the Resend Tools for sending emails. Use this tool to:

            - Compose and send HTML-formatted emails based on user input or specific requests.
            - Send emails to the given address, if the email has not been provided ask for it.
            - Include structured content, such as reports, summaries, or generated data.

            This tool is particularly useful for communication tasks requiring email-based delivery or automation.\
            """
        )

    logger.info(
        "Initializing combined knowledge base with sources and vector database."
    )
    knowledge_base = CombinedKnowledgeBase(
        sources=[
            PDFKnowledgeBase(
                path=citex_settings.knowledgebase_dir, reader=PDFReader(chunk=True)
            )
        ],
        vector_db=PgVector2(
            db_url=db_url,
            collection="llm_os_documents",
            embedder=OpenAIEmbedder(
                model=agent_settings.embedding_model, dimensions=1536
            ),
        ),
        num_documents=5,
    )

    logger.info("Loading the knowledge base (recreate=False).")
    return CitexGPT4Leader.build(
        team_members,
        id=agent_settings.Models.get_gpt_model(model_id),
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
                WORKFLOW: When the user sends a message, first **think** and determine if:
                    - You can answer using the tools available to you.
                    - You need to search the knowledge base (limited to 3 attempts with specific refinements).
                    - You need to search the internet if the knowledge base does not yield results.
                    - You need to delegate the task to a team member.
                    - You need to ask a clarifying question.\
                """
            ).strip(),
            (
                "IMPORTANT: **Prioritize** using **your available tools** before considering delegation. "
                "If no suitable tool is found, delegate the task to the appropriate team member."
            ),
            (
                "IMPORTANT: If the user asks about a topic, attempt to search your knowledge "
                "base using the `search_knowledge_base` tool **up to 3 times**. Each attempt "
                "should refine the query for better results. If no relevant results are found "
                "after 3 attempts, proceed to follow your WORKFLOW and skip the knowledge base."
            ),
            (
                "IMPORTANT: If you do not find relevant information in the knowledge base after "
                "3 attempts, use the `duckduckgo_search` tool to search the internet."
            ),
            (
                "IMPORTANT: If the user provides a YouTube link or URL, delegate the task to the "
                "YouTube Assistant team member to fetch and process the full captions of the video."
            ),
            (
                "IMPORTANT: If the user provides a Wikipedia link or URL or asks about Wikipedia, "
                "delegate the task to the Wikipedia Assistant team member."
            ),
            (
                "If the user asks to summarize the conversation, use the `get_chat_history` "
                "tool with None as the argument."
            ),
            "If the user's message is unclear, ask clarifying questions to get more information.",
            (
                "When gathering information, limit redundant searches and avoid repeated attempts "
                "on the same query or slight variations. Focus on concise, accurate queries."
            ),
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        # Add tools to the Assistant
        tools=tools,
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
