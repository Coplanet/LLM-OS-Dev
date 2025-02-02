from os import getenv, getuid

from agno.docker.app.fastapi import FastApi
from agno.docker.app.postgres import PgVectorDb
from agno.docker.app.streamlit import Streamlit
from agno.docker.resource.image import DockerImage
from agno.docker.resources import DockerResources

from workspace.settings import ws_settings

#
# -*- Resources for the Development Environment
#

# -*- Dev image
dev_image = DockerImage(
    name=f"{ws_settings.image_repo}/{ws_settings.image_name}",
    tag=ws_settings.dev_env,
    enabled=ws_settings.build_images,
    path=str(ws_settings.ws_root),
    push_image=False,
    use_cache=True,
    print_build_log=True,
    use_buildx=True,
    dockerfile=str(ws_settings.ws_root.joinpath("Dockerfile")),
    env_file=ws_settings.ws_root.joinpath(".env"),
    detach=False,
    network_mode="host",
    workspace_settings=ws_settings,
    buildargs={"DOCKER_UID": getuid()},
)

# -*- Dev database running on port 5432:5432
dev_db = PgVectorDb(
    name=f"{ws_settings.ws_name}-db",
    enabled=ws_settings.dev_db_enabled,
    pg_user="ai",
    pg_password="ai",
    pg_database="ai",
    # Connect to this db on port 5432
    host_port=5432,
    open_port=False,
)

# -*- Build container environment
container_env = {
    "RUNTIME_ENV": "dev",
    # Get the OpenAI API key from the local environment
    "OPENAI_API_KEY": getenv("OPENAI_API_KEY"),
    "PHI_MONITORING": "True",
    "PHI_API_KEY": getenv("PHI_API_KEY"),
    "COMPOSIO_API_KEY": getenv("COMPOSIO_API_KEY"),
    "COMPOSIO_LOGGING_LEVEL": getenv("COMPOSIO_LOGGING_LEVEL", "debug"),
    # Database configuration
    "DB_HOST": getenv("DB_HOST", dev_db.get_db_host()),
    "DB_PORT": getenv("DB_PORT", dev_db.get_db_port()),
    "DB_USER": getenv("DB_USER", dev_db.get_db_user()),
    "DB_PASS": getenv("DB_PASS", dev_db.get_db_password()),
    "DB_DATABASE": getenv("DB_DATABASE", dev_db.get_db_database()),
    # Wait for database to be available before starting the application
    "WAIT_FOR_DB": ws_settings.dev_db_enabled,
    # Migrate database on startup using alembic
    # "MIGRATE_DB": ws_settings.prd_db_enabled,
    "MIGRATE_DB": ws_settings.prd_db_enabled or True,
    # "INSTALL_REQUIREMENTS": True,
}

# -*- Streamlit running on port 8501:8501
dev_streamlit = Streamlit(
    name=f"{ws_settings.ws_name}-app",
    enabled=ws_settings.dev_app_enabled,
    image=dev_image,
    command="streamlit run app/Home.py",
    port_number=8501,
    open_port=False,
    debug_mode=True,
    mount_workspace=True,
    streamlit_server_headless=True,
    env_vars=container_env,
    force=True,
    use_cache=ws_settings.use_cache,
    # Read secrets from secrets/dev_app_secrets.yml
    secrets_file=ws_settings.ws_root.joinpath("workspace/secrets/dev_app_secrets.yml"),
    depends_on=[dev_db],
)

# -*- FastApi running on port 8000:8000
dev_fastapi = FastApi(
    name=f"{ws_settings.ws_name}-api",
    enabled=ws_settings.dev_api_enabled,
    image=dev_image,
    command="uvicorn api.main:app --reload",
    port_number=8000,
    open_port=False,
    debug_mode=True,
    mount_workspace=True,
    env_vars=container_env,
    use_cache=ws_settings.use_cache,
    # Read secrets from secrets/dev_app_secrets.yml
    secrets_file=ws_settings.ws_root.joinpath("workspace/secrets/dev_app_secrets.yml"),
    depends_on=[dev_db],
)

# -*- Dev DockerResources
dev_docker_resources = DockerResources(
    env=ws_settings.dev_env,
    network="host",  # ws_settings.ws_name,
    apps=[dev_db, dev_streamlit, dev_fastapi],
    workspace_settings=ws_settings,
)
