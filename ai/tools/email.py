from typing import Optional

from phi.tools.resend_tools import ResendTools


class EmailSenderTools(ResendTools):
    def __init__(
        self,
        api_key: Optional[str] = None,
        from_email: Optional[str] = None,
    ):
        super().__init__(api_key, from_email)
        self.name = "email_sender_tools"
