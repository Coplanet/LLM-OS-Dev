from textwrap import dedent

from phi.tools.exa import ExaTools

from helpers.tool_processor import process_tools
from workspace.settings import extra_settings

from .base import Agent, AgentConfig

agent = None
agent_name = "Patent Writer Agent"
available_tools = {
    ExaTools: {"name": "Exa", "kwargs": {"num_results": 5, "text_length_limit": 1000}}
}


def get_agent(config: AgentConfig = None):
    tools, _ = process_tools(agent_name, config, available_tools)

    # flake8: noqa: E501
    agent = Agent(
        name=agent_name,
        agent_config=config,
        tools=tools,
        save_output_to_file=extra_settings.scratch_dir / "{run_id}.md",
        role="Patent Writer",
        description=(
            "You are a specialized patent-writing assistant for scientific inventions."
        ),
        instructions=[
            dedent(
                """\
                You are a specialized patent-writing assistant for scientific inventions.
                Your task is to produce a complete, well-structured U.S. utility patent application for the invention:
                "[Insert a concise, high-level description of the scientific invention here, \
                e.g., 'a novel biodegradable polymer composition for sustained drug release']."

                Please follow these guidelines and format your output as a professional patent application
                intended for the U.S. Patent and Trademark Office (USPTO).
                The tone should be technical, formal, and legally oriented, avoiding marketing language and superlatives.
                Where appropriate, introduce exemplary embodiments and variations to fully illustrate the scope.

                IMPORTANT: Maintain consistent terminology and define specialized terms in the Detailed Description.
                IMPORTANT: Assume the reader is technically skilled in the relevant field but unfamiliar with this invention.
                IMPORTANT: Focus on technical clarity, legal soundness, and thoroughness.
                IMPORTANT: Present the invention so that it distinguishes itself clearly from existing solutions.
                IMPORTANT: Use the `Exa` tool to search for prior art and other relevant information and to write claims.
                IMPORTANT: Return the entire final patent document in a very professional and readable format.
                """
            ).strip(),
        ],
        expected_output=dedent(
            """
            The output should be a complete, well-structured U.S. utility patent application
            writen in professional and readable format as the following format/headline:

            1. Title of the Invention:
            - A concise, descriptive title indicating the nature of the invention.

            2. Cross-Reference to Related Applications (if any):
            - If related U.S. provisional or non-provisional applications exist, specify them. Otherwise, state 'Not Applicable.'

            3. Field of the Invention:
            - Briefly state the general technological or scientific field.

            4. Background of the Invention:
            - Describe the state of the art and the technical problem addressed.
            - Identify gaps or drawbacks in current solutions.
            - Avoid language that admits known references as prior art.

            5. Summary of the Invention:
            - Provide a succinct overview highlighting the invention’s key features and advantages.
            - Emphasize what is novel and non-obvious.

            6. Brief Description of the Drawings (if drawings are included or hypothesized):
            - List hypothetical figures with brief descriptions, referencing the aspects of the invention they illustrate.

            7. Detailed Description of the Invention:
            - Provide a thorough, technical explanation.
            - Include specific embodiments, materials, methods, parameters, and any experimental data (if available).
            - Reference figures as needed, making sure to describe how each element works.

            8. Claims:
            - Draft at least one independent claim encompassing the inventive concept broadly.
            - Include dependent claims that add specificity or preferred features.
            - Each claim should be a single sentence and follow standard patent claim writing conventions.

            9. Abstract:
            - Provide a concise (150 words or fewer) summary of the invention.

            return the patent:\
            """
        ).strip(),
        delegation_directives=[
            (
                f"Delegate any task that involves writing a patent document to the `{agent_name}`. "
                "This includes tasks where input data or insights are provided from any source, such as "
                "ArXiv documents, research papers, or user-provided information or any other data sources."
            ),
            (
                "IMPORTANT: Anything that is related to the **writing patents** or **writing patent** process, such as "
                "searching for prior art, writing claims, or any other task that is related to the patent writing process, "
                f"is delegated to the `{agent_name}`."
            ),
            (
                f"IMPORTANT: If the end goal is to write a patent, you should use the `{agent_name}` to write/writ the patent. "
                f"even if it means you need to use other tools to provide the necessary information to the `{agent_name}`."
            ),
            (
                f"IMPORTANT: return the output of the `{agent_name}` without any additional formatting or "
                "comments or contribution to the output."
            ),
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
