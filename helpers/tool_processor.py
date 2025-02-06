from typing import Any, Dict, List, Optional, Tuple, Union

from agno.tools import Toolkit
from agno.tools.function import Function

from ai.agents.settings import AgentConfig

from .log import logger


def _process_tools_dict(
    agent_name: str,
    config: Optional[AgentConfig],
    available_tools: Dict[Toolkit.__class__, dict],
) -> Tuple[List[Toolkit], List[str]]:
    tools = []
    extra_instructions = []

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


def _process_tools_list(
    agent_name: str,
    config: Optional[AgentConfig],
    available_tools: List[Dict[str, Any]],
) -> Tuple[List[Function], List[str]]:
    tools = []
    extra_instructions = []

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


def process_tools(
    agent_name: str,
    config: Optional[AgentConfig],
    available_tools: Union[List[Dict[str, Any]], Dict[Toolkit.__class__, dict]],
) -> Tuple[List[Toolkit], List[str]]:

    if not config:
        config = AgentConfig.empty()

    if config.is_empty:
        config.tools = available_tools

    tools: list[Toolkit] = None
    extra_instructions: list[str] = None

    if isinstance(available_tools, dict):
        tools, extra_instructions = _process_tools_dict(
            agent_name, config, available_tools
        )

    elif isinstance(available_tools, list):
        tools, extra_instructions = _process_tools_list(
            agent_name, config, available_tools
        )

    else:
        raise ValueError(
            "Invalid type for available_tools. Expected a dictionary or list."
        )

    if not tools:
        return None, []

    if not extra_instructions:
        return tools, []

    return tools, extra_instructions
