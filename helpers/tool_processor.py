from typing import Any, Dict, List, Optional, Tuple

from phi.tools import Toolkit
from phi.tools.function import Function

from ai.agents.settings import AgentConfig

from .log import logger


def process_tools(
    agent_name: str,
    config: Optional[AgentConfig],
    available_tools: Dict[Toolkit.__class__, dict],
) -> Tuple[List[Toolkit], List[str]]:
    tools = []
    extra_instructions = []

    if not config:
        config = AgentConfig.empty()

    if config.is_empty:
        config.tools = available_tools

    for tool in config.tools or {}:
        if tool in available_tools:
            logger.debug(
                f"Adding '{tool.__name__}' tool to '{agent_name}' with full functionality."
            )

            if "instance" in available_tools[tool]:
                tools.append(available_tools[tool]["instance"])

            else:
                tools.append(tool(**(available_tools[tool].get("kwargs", {}))))

            if "extra_instructions" in available_tools[tool]:
                ei = available_tools[tool].get("extra_instructions")
                if not isinstance(ei, str):
                    logger.warning(
                        f"Extra instructions for '{tool.__name__}' in '{agent_name}' is not a string; "
                        f"skipping: '{ei}'"
                    )
                    continue
                extra_instructions.append(ei)

        else:
            logger.warning(
                f"Tool '{tool.__name__}' not found in available tools for '{agent_name}'."
            )

    return tools, extra_instructions


def process_composio_tools(
    agent_name: str,
    config: Optional[AgentConfig],
    available_tools: List[Dict[str, Any]],
) -> Tuple[List[Function], List[str]]:
    tools = []
    extra_instructions = []

    if not config:
        config = AgentConfig.empty()

    if config.is_empty:
        config.tools = available_tools

    available_tools_manifest = {}

    for tool in available_tools:
        if not isinstance(tool, dict):
            raise ValueError(
                f"Tool '{tool.__name__}' is not a dictionary in package '{agent_name}'."
            )

        available_tools_manifest[tool.get("name")] = tool

    for tool in config.tools or {}:
        if isinstance(tool, dict):
            tool = tool.get("name")

        if tool in available_tools_manifest:
            logger.debug(
                f"Adding '{tool}' tool to '{agent_name}' with full functionality."
            )

            logger.debug(
                f"Adding '{tool}' tool to '{agent_name}' with full functionality."
            )

            if "instance" not in available_tools_manifest[tool]:
                raise ValueError(
                    f"Tool '{tool}' does not have an instance in package '{agent_name}'."
                )

            tools.append(available_tools_manifest[tool]["instance"])

            if "extra_instructions" in available_tools_manifest[tool]:
                ei = available_tools_manifest[tool].get("extra_instructions")
                if not isinstance(ei, str):
                    logger.warning(
                        f"Extra instructions for '{tool.__name__}' in '{agent_name}' is not a string; "
                        f"skipping: '{ei}'"
                    )
                    continue
                extra_instructions.append(ei)

        else:
            logger.warning(
                f"Tool '{tool.__name__}' not found in available tools for '{agent_name}'."
            )

    return tools, extra_instructions
