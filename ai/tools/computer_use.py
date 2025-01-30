from phi.agent import Agent
from phi.tools import Toolkit

from db.session import get_db_context
from db.tables.user_config import UserNextOp


class ComputerUseTools(Toolkit):
    def __init__(self):
        super().__init__()
        self.name = "computer_use_tools"
        self.register(self.user_wants_to_use_computer)

    def user_wants_to_use_computer(self, agent: Agent):
        """
        Use this method when the user wants the "computer use" interface.

        Returns:
            str: A message indicating that the computer use's interface is provided.
        """

        with get_db_context() as db:
            UserNextOp.save_op(
                db,
                agent.session_id,
                UserNextOp.COMPUTER_USE,
            )

        return "The UI will guide the user to the computer use's interface."
