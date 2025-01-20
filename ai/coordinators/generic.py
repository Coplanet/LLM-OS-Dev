from textwrap import dedent
from typing import Dict, Optional

from phi.agent import AgentMemory
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.memory.db.postgres import PgMemoryDb
from phi.storage.agent.postgres import PgAgentStorage
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.tools.calculator import Calculator
from phi.tools.exa import ExaTools
from phi.tools.wikipedia import WikipediaTools
from phi.tools.yfinance import YFinanceTools
from phi.tools.youtube_tools import YouTubeTools
from phi.vectordb.pgvector import PgVector2

from ai.agents import funny, journal, linkedin_content_generator, patent_writer, python
from ai.agents.base import AgentTeam, Provider
from ai.agents.settings import AgentConfig, agent_settings
from ai.tools.email import EmailSenderTools
from ai.tools.file import FileIOTools
from ai.tools.stability import Stability
from ai.tools.website_crawler import WebSiteCrawlerTools
from db.session import db_url
from db.settings import db_settings
from helpers.log import logger
from helpers.tool_processor import process_tools
from workspace.settings import extra_settings

from .base import Coordinator
from .composio_tools import COMPOSIO_ACTIONS

agent = None
agent_name = "Coordinator"
available_tools = [
    {
        "order": 100,
        "instance": Stability(),
        "name": "Stability",
        "extra_instructions": dedent(
            """\
            Use the Stability tool to generate/edit/recolor images.

            <critical_notes_for_image_tools>
            **CRITICAL NOTES: NEVER DIVERGE FROM THESE NOTES WHEN USING THE IMAGE TOOLS!**
            - **NEVER** try provide to provide the image's data by yourself,
            the image link will be generated from the Stability tool.
            and provided to the user in different flow.
            **CRITICAL**: If based on the image and user description, you think that the output image after editing won't be good (specially with `add_feature_or_change_accurately` tool), warn the user about it and if the user presist to edit the image with given prompt, you continue with the task.
            - Use `add_feature_or_change_accurately` tool to accurately add/remove/change something in the image.
                - **IMPORTANT**: Never use this tool to change the color of the image.
                - **IMPORTANT**: Alternative to `search_and_replace` is `add_feature_or_change_accurately` tool that can accurately edit the image.
                - **IMPORTANT**: When this model return a json that indicates it needs to create a mask, DO NOT switch to different the tool, the UI flow that you are operating in, knows how to handle the mask.
                - **IMPORTANT**: After capturing the mask, DO NOT switch to different the tool/function, the UI flow that you are operating in, knows how to handle the mask internally.
                - **IMPORTANT**: After running the `add_feature_or_change_accurately` tool for the first time (getting the mask), The UI will resend the original prompt and you have to send it again to `add_feature_or_change_accurately` tool.
            - Use the `search_and_recolor` tool to change entire or part of the image ONLY and nothing else.
            - Use the `remove_background` tool to remove the background of the image (only to remove the background).
            - Use the `outpaint` tool to add extra pixels to the image.
            - Use the `search_and_replace` tool to change the color of the image and explicitly asked for replacing an object in the image.
                - **IMPORTANT**: Never use `search_and_replace` tool to change the color of the image without explicitly asked for replacing an object in the image.
                - **IMPORTANT**: Alternative to `search_and_replace` is `add_feature_or_change_accurately` tool that can accurately edit the image.
            - Use the `create_image` tool to create a new image based on the provided prompt (**NEVER** use this tool to edit the image).
            </critical_notes_for_image_tools>
            """  # flake8: noqa: E501
        ).strip(),
        "icon": "fa-solid fa-image",
    },
    {
        "order": 200,
        "instance": YouTubeTools(),
        "name": "YouTube",
        "extra_instructions": dedent(
            """\
            Use the YouTube tool to search for and analyze YouTube videos.

            To analyze YouTube videos, delegate to the Youtube tool. Youtube tool can:
            - Extract video captions from YouTube URLs (for transcript requests)
            - Extract and analyze video metadata and captions for analysis requests
            - Provide video summaries and answer questions about content

            IMPORTANT: **For transcript-only** requests, the Youtube tool will return raw captions without analysis. **DO NOT modify or process this response**.
            NOTE: The Youtube tool works best with videos that have captions available.
            Do NOT delegate non-YouTube video analysis tasks to Youtube tool agent.
            """
        ).strip(),
        "icon": "fa-brands fa-youtube",
    },
    {
        "order": 300,
        "instance": ArxivToolkit(),
        "name": "Arxiv",
        "icon": "fa-solid fa-book-open",
    },
    {
        "order": 400,
        "instance": WikipediaTools(),
        "name": "Wikipedia",
        "icon": "fab fa-wikipedia-w",
    },
    {
        "order": 500,
        "instance": YFinanceTools(
            stock_price=True,
            company_info=True,
            analyst_recommendations=True,
            company_news=True,
        ),
        "name": "YFinance",
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
        "icon": "fa-solid fa-chart-line",
    },
    {
        "order": 600,
        "instance": ExaTools(num_results=5, text_length_limit=1000),
        "name": "Search (Exa)",
        "extra_instructions": dedent(
            """\
            Leverage the Exa Search tool for quick internet searches, such as finding \
                up-to-date information,
            verifying facts, or answering questions beyond the scope of the knowledge base.
            Use this tool when a direct query requires additional context or when you need to retrieve concise
            and relevant information (limited to 3 results per search).\
            """
        ).strip(),
        "icon": "fa-solid fa-magnifying-glass",
    },
    {
        "order": 700,
        "instance": EmailSenderTools(
            api_key=extra_settings.resend_api_key,
            from_email=extra_settings.resend_email_address,
        ),
        "name": "Email Sender",
        "extra_instructions": dedent(
            """\
            Utilize the Email Sender Tools exclusively for sending emails. This tool is designed to:

            - Sending emails to the given address, if the email has not been provided ask for it.

            This tool is particularly useful for communication tasks requiring email-based delivery or automation.\
            """
        ).strip(),
        "icon": "fa-solid fa-envelope",
    },
    {
        "order": 800,
        "instance": FileIOTools(base_dir=extra_settings.scratch_dir),
        "name": "File IO",
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
        "icon": "fa-solid fa-file-alt",
    },
    {
        "order": 900,
        "instance": WebSiteCrawlerTools(),
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
        "icon": "fa-solid fa-globe",
    },
    {
        "order": 1000,
        "instance": Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
        ),
        "name": "Calculator",
        "extra_instructions": dedent(
            """\
            Use the Calculator tool for precise and complex mathematical operations, including addition,
            subtraction, multiplication, division, exponentiation, factorials, checking if a number is prime,
            and calculating square roots. This tool is ideal for mathematical queries, computations,
            or when the user needs help solving equations or understanding numeric concepts.\
            """
        ).strip(),
        "icon": "fa-solid fa-calculator",
    },
]

for group, details in COMPOSIO_ACTIONS.items():
    for order, instance in enumerate(
        agent_settings.composio_tools.get_tools(actions=details["actions"])
    ):
        name = details["name"]
        available_tools.append(
            {
                "group": name,
                "order": 500 + order + 1,
                "instance": instance,
                "name": instance.name,
                "icon": details["icon"],
                "default_status": "disabled",
                # "extra_instructions": "Use `{}` tool for {}".format(
                #     instance.name, instance.functions["function_template"].description,
                # ),
            }
        )


available_tools = sorted(available_tools, key=lambda x: x["order"])


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
        config.model = Provider.OpenAI.value
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
    conditional_agent_enable(journal)
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

    instructions = [
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
            "VERY IMPORTANT: When an agent/tool didn't return anything **when it supposed to return something**, retry and **enforce** the agent to return the result. "
            "Make sure to **think** before retrying and enforce the agent to return the result if the empty result is accepted or not. "
            "**NOTE** If the agent returns **error message**, don't retry and just return the error message."
        ),
        (
            "VERY IMPORTANT: Remember, **as a leader**, your primary role is to delegate effectively and "
            "empower your agent team. Before diving into any task yourself, consider whether it can be "
            "assigned to your team members."
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
        dedent(
            """\
            TASK DECOMPOSITION: For complex tasks that require multiple steps:
            1. Break down the task into clear, sequential sub-tasks
            2. Execute each sub-task independently using the appropriate tool or team member
            3. Use the output from each step as input for the next step
            4. Maintain the specific role boundaries of each team member

            For example:
            - If a task requires gathering information first, complete that step before processing
            - If multiple sources need to be analyzed, gather all sources before synthesis
            - If content needs to be transformed (e.g., into an agent), gather all inputs before delegation

            DELEGATION RULES:
            1. Each team member has a specific role - do not ask them to perform tasks outside their expertise
            2. Gather all necessary inputs before delegating a creative or synthesis task
            3. When multiple team members are needed, coordinate their outputs in sequence
            4. Always validate the output of each step before proceeding to the next\
            """
        ).strip(),
        (
            "VERY IMPORTANT: When handling multi-step tasks, explicitly state your step-by-step plan "
            "before execution and ensure each team member focuses solely on their specialized role."
        ),
        (
            "CRITICAL: Never ask a team member to perform tasks outside their designated function. "
            "For example, research agents should only gather and analyze information, while creative "
            "agents should only work with prepared inputs."
        ),
        (
            "**IMPORTANT**: When sending emails, always use HTML format. if the input is not HTML "
            "(e.g. markdown), convert it to HTML before sending it."
        ),
        "If no image is provided, and user asks for an image related task, ask the user to provide one.",
    ]

    description = dedent(
        """\
        You are the most advanced AI system in the world called `LLM OS`.
        You have access to a set of tools and a team of AI Assistants at your disposal.
        Your goal is to assist the user in the best way possible.
        """
    ).strip()

    if (
        config.provider == Provider.OpenAI.value
        and config.model_id == "gpt-4o-audio-preview"
    ):
        instructions += [
            "**RED-LINE CRITICAL:** Always respond to the user in voice format.",
            "**RED-LINE CRITICAL:** Your input is in text format. but your response should be in voice format.",
            "**RED-LINE CRITICAL:** Your response should be in voice format. Do not respond in text format.",
        ]
        description += (
            "\nYou are a helpful assistant. "
            "You are able to understand and respond to the user's voice input."
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
        # Store the memories and summary in a database
        memory=AgentMemory(
            db=PgMemoryDb(table_name="agent_memory", db_url=db_settings.get_db_url()),
            create_user_memories=True,
            create_session_summary=True,
        ),
        introduction=dedent(
            """\
            Hi, I'm your LLM OS.
            I have access to a set of tools and AI Assistants to assist you.
            Lets get started!\
            """
        ).strip(),
        description=dedent(
            """\
            You are the most advanced AI system in the world called `LLM OS`.
            You have access to a set of tools and a team of AI Assistants at your disposal.
            Your goal is to assist the user in the best way possible.\
            """
        ).strip(),
        instructions=instructions,
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
