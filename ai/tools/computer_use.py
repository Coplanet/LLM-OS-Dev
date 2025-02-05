from agno.agent import Agent
from agno.tools import Toolkit

from db.session import get_db_context
from db.tables.user_config import UserNextOp


class ComputerUseTools(Toolkit):
    def __init__(self):
        super().__init__()
        self.name = "computer_use_tools"
        self.register(self.user_wants_to_use_computer)

    def user_wants_to_use_computer(self, agent: Agent, platform: str):
        """
        Use this method when the user wants the "computer use" interface.

        Args:
            platform: The platform to use for the computer use's interface. \
                Detect the platform from the user's request. \
                Default is "gemini". and accepted values are "gemini", "anthropic", "google".

        Returns:
            str: A message indicating that the computer use's interface is provided.
        """

        platform = {
            "gemini": "gemini",
            "anthropic": "anthropic",
            "google": "gemini",
        }.get(platform)

        if not isinstance(platform, str) or platform not in ["gemini", "anthropic"]:
            return "Currently we only support Gemini and Anthropic platforms."

        with get_db_context() as db:
            UserNextOp.save_op(
                db,
                agent.session_id,
                UserNextOp.COMPUTER_USE,
                value={"platform": platform},
            )

        return "The UI will guide the user to the computer use's interface."
