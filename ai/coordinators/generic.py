from textwrap import dedent
from typing import Dict, Optional

from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.storage.agent.postgres import PgAgentStorage
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.vectordb.pgvector import PgVector2

from ai.agents import (
    arxiv,
    funny,
    github,
    google_calender,
    journal,
    linkedin_content_generator,
    patent_writer,
    python,
    wikipedia,
    youtube,
)
from ai.agents.base import AgentTeam
from ai.agents.settings import AgentConfig, agent_settings
from ai.tools.email import EmailSenderTools
from ai.tools.file import FileIOTools
from ai.tools.website_crawler import WebSiteCrawlerTools
from db.session import db_url
from db.settings import db_settings
from helpers.log import logger
from helpers.tool_processor import process_tools
from workspace.settings import extra_settings

from .base import Coordinator

agent = None
agent_name = "Coordinator"
available_tools = {
    Calculator: {
        "name": "Calculator",
        "kwargs": {
            "add": True,
            "subtract": True,
            "multiply": True,
            "divide": True,
            "exponentiate": True,
            "factorial": True,
            "is_prime": True,
            "square_root": True,
        },
        "extra_instructions": dedent(
            """\
            Use the Calculator tool for precise and complex mathematical operations, including addition,
            subtraction, multiplication, division, exponentiation, factorials, checking if a number is prime,
            and calculating square roots. This tool is ideal for mathematical queries, computations,
            or when the user needs help solving equations or understanding numeric concepts.\
            """
        ).strip(),
    },
    DuckDuckGo: {
        "name": "DuckDuckGo",
        "kwargs": {"fixed_max_results": 3},
        "extra_instructions": dedent(
            """\
            Leverage the DuckDuckGo Search tool for quick internet searches, such as finding \
                up-to-date information,
            verifying facts, or answering questions beyond the scope of the knowledge base.
            Use this tool when a direct query requires additional context or when you need to retrieve concise
            and relevant information (limited to 3 results per search).\
            """
        ).strip(),
    },
    YFinanceTools: {
        "name": "YFinance",
        "kwargs": {
            "stock_price": True,
            "company_info": True,
            "analyst_recommendations": True,
            "company_news": True,
        },
        "extra_instructions": dedent(
            """\
            Utilize YFinance tools for financial and stock-related queries. This includes:

            - Stock Price: Retrieve real-time or historical stock prices.
            - Company Info: Fetch detailed company profiles or background.
            - Analyst Recommendations: Provide insights into stock recommendations from financial analysts.
            - Company News: Share recent news articles or updates related to the company.

            These tools are ideal for answering finance-related questions or assisting with investment decisions.\
            """
        ).strip(),
    },
    FileIOTools: {
        "name": "File IO",
        "kwargs": {"base_dir": extra_settings.scratch_dir},
        "extra_instructions": dedent(
            """\
            Use the File IO Tools for managing files in the working directory. Specific use cases include:

            - Read Files: Open and read content from files when the user uploads or references one.
            - Save Files: Store data, results, or responses in a file upon request.
            - List Files: Display the contents of the working directory for easy navigation.

            This tool is helpful for file manipulation tasks, processing user-uploaded data, or \
            saving generated content for future use.\
            """
        ).strip(),
    },
    EmailSenderTools: {
        "name": "Email Sender",
        "kwargs": {
            "api_key": extra_settings.resend_api_key,
            "from_email": "onboarding@resend.dev",
        },
        "extra_instructions": dedent(
            """\
            Employ the Email Sender Tools for sending emails. Use this tool to:

            - Compose and send HTML-formatted emails based on user input or specific requests.
            - Send emails to the given address, if the email has not been provided ask for it.
            - Include structured content, such as reports, summaries, or generated data.

            This tool is particularly useful for communication tasks requiring email-based delivery or automation.\
            """
        ).strip(),
    },
    WebSiteCrawlerTools: {
        "name": "Website Crawler",
        "extra_instructions": dedent(
            """\
            Use the Website Crawler Tools to parse the content of a website and add it to the knowledge base.
            This tool is ideal for integrating external web content into the system for future reference or analysis.
            Use it when the user provides a website URL or requests detailed insights from a web page.
            Ensure the content is relevant and valuable before adding it to the knowledge base to maintain its \
                quality and relevance.
            """
        ).strip(),
    },
}


def get_coordinator(
    config: Optional[AgentConfig] = None,
    team_config: Dict[str, AgentConfig] = {},
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    if config is None:
        config = AgentConfig.empty()

    if config.is_empty:
        config.model = "OpenAI"
        config.model_id = "gpt-4o"
        config.enabled = True
        config.temperature = 0
        config.max_tokens = agent_settings.default_max_completion_tokens
        config.tools = available_tools

    team_members = AgentTeam()

    def conditional_agent_enable(pkg):
        config: AgentConfig = team_config.get(pkg.agent_name)

        if not config or config.is_empty or config.enabled:
            pkg.agent = pkg.get_agent(
                config if config and not config.is_empty else None
            )

        if pkg.agent:
            logger.debug("Activating %s", pkg.agent_name)
            team_members.activate(pkg.agent)

        else:
            pkg.agent = None
            logger.debug("DEACTICATING %s", pkg.agent_name)

    conditional_agent_enable(python)
    conditional_agent_enable(youtube)
    conditional_agent_enable(arxiv)
    conditional_agent_enable(journal)
    conditional_agent_enable(wikipedia)
    conditional_agent_enable(github)
    conditional_agent_enable(google_calender)
    conditional_agent_enable(patent_writer)
    conditional_agent_enable(funny)
    conditional_agent_enable(linkedin_content_generator)

    tools, extra_instructions = process_tools(agent_name, config, available_tools)

    logger.debug(
        "Initializing combined knowledge base with sources and vector database."
    )
    knowledge_base = CombinedKnowledgeBase(
        sources=[
            PDFKnowledgeBase(
                path=extra_settings.knowledgebase_dir, reader=PDFReader(chunk=True)
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

    logger.debug("Loading the knowledge base (recreate=False).")

    # flake8: noqa: E501
    agent = Coordinator.build(
        team_members,
        model=config.get_model,
        name=agent_name,
        role="Lead the team to complete the task",
        # Add tools to the Assistant
        tools=tools,
        # Introduce knowledge base to the leader
        knowledge_base=knowledge_base,
        # Set addication context to the system's prompt
        additional_context=extra_instructions,
        # Inject some app related items
        run_id=run_id,
        user_id=user_id,
        session_id=session_id,
        storage=PgAgentStorage(
            table_name="agent_sessions", db_url=db_settings.get_db_url()
        ),
        introduction=dedent(
            """\
                Hi, I'm your LLM OS.
                I have access to a set of tools and AI Assistants to assist you.
                Lets get started!\
                """
        ),
        description=dedent(
            """\
                You are the most advanced AI system in the world called ` LLM OS`.
                You have access to a set of tools and a team of AI Assistants at your disposal.
                Your goal is to assist the user in the best way possible.\
                """
        ),
        instructions=[
            dedent(
                """\
                WORKFLOW: When the user sends a message, first **think** and determine if:
                    - You need to search the knowledge base (limited to 3 attempts with specific refinements).
                    - You need to answer using the tools available to you.
                    - You need to search the internet if the knowledge base does not yield results.
                    - You need to delegate the task to a team member.
                    - You need to ask a clarifying question.

                After you conclude your thought process, **respond** to the user with the appropriate action in the given order above.\
                """
            ).strip(),
            (
                "IMPORTANT: If the user asks about a topic, **FIRST** attempt to search your knowledge "
                "base using the `search_knowledge_base` tool **up to 3 times**. Each attempt "
                "should refine the query for better results. If no relevant results are found "
                "after 3 attempts, proceed to follow your WORKFLOW and skip the knowledge base."
            ),
            (
                "IMPORTANT: **Prioritize** using **your available tools** before considering delegation. "
                "If no suitable tool is found, delegate the task to the appropriate team member."
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
        # This setting adds a tool to search the knowledge base for information
        search_knowledge=True,
        # This setting adds a tool to get chat history
        read_chat_history=True,
        # Add the previous chat history to the messages sent to the Model.
        add_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_responses=6,
    )
    agent.read_from_storage()
    return agent


__all__ = ["get_coordinator", "agent_name", "AgentConfig"]
