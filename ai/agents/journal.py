from textwrap import dedent

from phi.tools.exa import ExaTools

from workspace.settings import extra_settings

from .base import Agent, AgentConfig

agent_name = "Journal Writer Agent"


def get_agent(config: AgentConfig = None):
    return Agent(
        name=agent_name,
        agent_config=config,
        role="Write a research report on a given topic",
        tools=[ExaTools(num_results=5, text_length_limit=1000)],
        save_output_to_file=extra_settings.scratch_dir / "{run_id}.md",
        description="You are a Senior Report Writer tasked with writing a cover story research report.",
        instructions=[
            "For a given topic, use the `search_exa` to get the top 10 search results.",
            (
                "Carefully read the results and generate a final - NYT cover story worthy "
                "report in the format provided below."
            ),
            "Make your report engaging, informative, and well-structured.",
            "Remember: you are writing for the advanced journals, so the quality of the report is important.",
        ],
        expected_output=dedent(
            """\
        An engaging, informative, and well-structured report in the following format:

        ## Title

        - **Overview** Brief introduction of the topic.
        - **Importance** Why is this topic significant now?

        ### Section 1
        - **Detail 1**
        - **Detail 2**

        ### Section 2
        - **Detail 1**
        - **Detail 2**

        ## Conclusion
        - **Summary of report:** Recap of the key findings from the report.
        - **Implications:** What these findings mean for the future.

        ## References
        - [Reference 1](Link to Source)
        - [Reference 2](Link to Source)\
        """
        ).strip(),
        delegation_directives=[
            (
                "To write a research report, delegate the task to the `Research Assistant`. "
                "Return the report in the <report_format> to the user as is, without any "
                "additional text like 'here is the report'."
            ),
        ],
    )


__all__ = ["get_agent", "agent_name"]
