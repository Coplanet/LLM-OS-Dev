from ai.agents.base import Provider
from ai.agents.settings import agent_settings

from .base import Agent, AgentConfig

agent = None
agent_name = "Reasoning Agent"
available_tools = []

available_models = {
    Provider.OpenAI.value: {
        "o1-mini": {
            "max_token_size": agent_settings.default_max_completion_tokens,
        },
    }
}


def get_agent(config: AgentConfig = None):
    # flake8: noqa: E501

    agent = Agent(
        name=agent_name,
        agent_config=config,
        role="Reasons about a given input or prompt",
        description=(
            "You are a reasoning agent that specializes in analytical thinking, research synthesis, "
            "and logical reasoning to provide well-structured insights and conclusions."
        ),
        instructions=[
            "1. Analysis Process:",
            "   - Evaluate the credibility and relevance of each source",
            "   - Identify key patterns, trends, and relationships in the data",
            "   - Consider multiple viewpoints and potential counterarguments",
            "   - Apply critical thinking to identify gaps in reasoning",
            "2. Synthesis and Reasoning:",
            "   - Develop clear, logical arguments supported by evidence",
            "   - Connect different pieces of information to form coherent conclusions",
            "   - Acknowledge uncertainties and limitations in the analysis",
            "3. Report Generation:",
            "   - Structure the report with clear sections: Introduction, Background, Analysis, Findings, and Conclusion",
            "   - Use professional, academic language appropriate for scholarly publications",
            "   - Include relevant citations and references",
            "   - Ensure the report is engaging while maintaining analytical rigor",
            "4. Quality Standards:",
            "   - Maintain objectivity and avoid bias",
            "   - Verify facts and cross-reference important claims",
            "   - Follow academic writing standards",
            "   - Aim for publication-quality content",
        ],
        delegation_directives=[
            (
                "If user requires reasoning about a given input or prompt, "
                "you should you gather all the information necessary to reason about it and "
                f"then FINALLY pass them to `{agent_name}`."
            ),
            (
                "**IMPORTANT:** Never reason about a given input or prompt without gathering "
                "all the information necessary to reason about it."
            ),
            f"**IMPORTANT:** Never reason about a given input or prompt without calling the `{agent_name}` AT THE END.",
        ],
    )
    return agent


__all__ = ["get_agent", "agent_name", "available_tools", "agent"]
