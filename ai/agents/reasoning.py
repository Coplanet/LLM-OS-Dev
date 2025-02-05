from ai.agents.base import Provider
from ai.agents.settings import agent_settings
from app.models import SupportStrength, SupportTypes

from .base import Agent, AgentConfig

agent = None
agent_name = "Reasoning Agent"
available_tools = []

available_models = {
    Provider.OpenAI.value: {
        "o1-mini": {
            "kwargs": {
                "max_completion_tokens": agent_settings.default_max_completion_tokens,
            },
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
            },
        },
        "o3-mini": {
            "kwargs": {
                "max_completion_tokens": 100_000,
            },
            "supports": {
                SupportTypes.TextIn: SupportStrength.Full,
                SupportTypes.TextOut: SupportStrength.Full,
                SupportTypes.FunctionCalling: SupportStrength.Full,
            },
        },
    }
}

default_model_id = "o3-mini"
default_model_type = Provider.OpenAI.value

default_model_config = {
    "model_type": default_model_type,
    "model_id": default_model_id,
    "model_kwargs": available_models[default_model_type][default_model_id]["kwargs"],
    "temperature": 1,  # this is the only temperature for o1-mini
    "enabled": True,
}


def get_agent(config: AgentConfig = None):
    # flake8: noqa: E501

    agent = Agent(
        name=agent_name,
        agent_config=config,
        available_models=available_models,
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
                f"You MUST ALWAYS delegate ANY and ALL reasoning tasks to `{agent_name}`. This is a strict requirement. "
                "Whenever you need to analyze, evaluate, or draw conclusions about ANY information, "
                f"you MUST gather the relevant information and delegate to `{agent_name}`."
            ),
            (
                "**ABSOLUTELY REQUIRED:** You are NOT PERMITTED to perform reasoning tasks yourself. "
                "This includes but is not limited to: analyzing data, evaluating options, drawing conclusions, "
                f"making comparisons, or synthesizing information. ALL such tasks MUST be delegated to `{agent_name}`."
            ),
            (
                "**CRITICAL WORKFLOW:**\n"
                "1. When reasoning is needed, gather ALL relevant information first\n"
                f"2. ALWAYS delegate the reasoning task to `{agent_name}`\n"
                f"3. Use `{agent_name}` output as your reasoning conclusion"
            ),
        ],
    )
    return agent


__all__ = [
    "get_agent",
    "agent_name",
    "available_tools",
    "agent",
    "available_models",
    "default_model_id",
    "default_model_type",
]
