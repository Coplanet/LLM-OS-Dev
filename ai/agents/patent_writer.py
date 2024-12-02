from textwrap import dedent

from phi.tools.exa import ExaTools

from workspace.settings import extra_settings

from .base import Agent

agent = Agent(
    name="Patent Writer Agent",
    tools=[ExaTools(num_results=5, text_length_limit=1000)],
    save_output_to_file=extra_settings.scratch_dir / "{run_id}.md",
    role="Draft a patent document for a specified invention",
).register_or_load(
    default_agent_config={
        "description": (
            "An AI system designed to assist in drafting patent applications, "
            "ensuring clarity and adherence to patent office standards."
        ),
        "instructions": [
            "Use the `search_exa` tool to gather relevant prior art and technical information.",
            "Analyze the collected data to draft a comprehensive patent document.",
            "Ensure the document is clear, precise, and follows the standard patent format.",
            "Focus on highlighting the novelty and inventive step of the invention.",
        ],
        "expected_output": dedent(
            """\
        A structured patent document comprising the following sections:

        ## Title

        - **Abstract**: A brief summary of the invention, highlighting its purpose and key features.

        ## Background

        - **Field of the Invention**: The technical area to which the invention pertains.
        - **Description of Related Art**: Discussion of prior art and the limitations addressed by the invention.

        ## Summary

        - **Summary of the Invention**: Overview of the invention and its advantages.

        ## Detailed Description

        - **Detailed Description**: In-depth explanation of the invention, including embodiments and examples.

        ## Claims

        - **Claims**: Specific legal claims defining the scope of the invention's protection.

        ## Drawings (if applicable)

        - **Figure Descriptions**: Brief descriptions of any accompanying drawings.

        ## References

        - [Reference 1](Link to Source)
        - [Reference 2](Link to Source)\
        """
        ).strip(),
        "delegation_directives": [
            (
                "Delegate the task of drafting a patent document to the `Patent Writer Agent`. "
                "Provide the document in the specified format directly to the user without additional commentary."
            ),
        ],
    },
)


__all__ = ["agent"]
