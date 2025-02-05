import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from agno.workspace.settings import WorkspaceSettings

#
# -*- Define workspace settings using a WorkspaceSettings object
# these values can also be set using environment variables or a .env file
#

ROOT_DIR = Path(__file__).parent.parent.resolve()

ws_settings = WorkspaceSettings(
    # Path to the workspace root
    ws_root=ROOT_DIR,
    # Workspace name: used for naming cloud resources
    ws_name="llmos-plus",
    # -*- Development env settings
    dev_env="dev",
    # -*- Development Apps
    dev_app_enabled=True,
    dev_api_enabled=True,
    dev_db_enabled=True,
    # -*- Production env settings
    prd_env="prd",
    # -*- Production Apps
    prd_app_enabled=True,
    prd_api_enabled=True,
    prd_db_enabled=True,
    # -*- AWS settings
    # Region for AWS resources
    aws_region="us-east-1",
    # Availability Zones for AWS resources
    aws_az1="us-east-1b",
    aws_az2="us-east-1c",
    aws_profile="PowerUserAccess-577638388865",
    # Subnet IDs in the aws_region
    subnet_ids=[
        "subnet-064e43373507fc6a1",
        "subnet-02b914b335534c283",
        "subnet-0ccc7cbf93d781e11",
    ],
    # -*- Image Settings
    # Name of the image
    image_name="agent-app",
    # Repository for the image
    image_repo="577638388865.dkr.ecr.us-east-1.amazonaws.com/llm-os",
)


class Settings:
    # Path to the workspace root
    ws_root: Path = ROOT_DIR
    ollama_host: Optional[str] = os.getenv("OLLAMA_HOST", "localhost")
    gpt_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    resend_api_key: Optional[str] = os.getenv("RESEND_API_KEY", None)
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY", None)
    resend_email_address: Optional[str] = os.getenv(
        "RESEND_EMAIL_ADDRESS", "info@test.coplanet.com"
    )
    composio_api_key: Optional[str] = os.getenv("COMPOSIO_API_KEY", None)
    secret_key: Optional[str] = os.getenv("SECRET_KEY", None)
    domain: Optional[str] = os.getenv("DOMAIN", "localhost:8501")
    domain_scheme: Optional[str] = os.getenv("DOMAIN_SCHEME", "http")
    proxies: Optional[Dict[str, Any]] = json.loads(os.getenv("PROXIES", "{}")) or None
    anthropic_computer_url_link: Optional[str] = os.getenv(
        "ANTHROPIC_COMPUTER_URL_LINK", None
    )
    gemini_computer_url_link: Optional[str] = os.getenv(
        "GEMINI_COMPUTER_URL_LINK", None
    )

    @property
    def scratch_dir(self):
        return self.ws_root.joinpath("scratch")

    @property
    def knowledgebase_dir(self):
        return self.ws_root.joinpath("knowledgebase")

    def get_redirect_url(self, params: dict):
        return (
            f"{self.domain_scheme}://{self.domain}/?callback=true&{urlencode(params)}"
        )


extra_settings = Settings()
