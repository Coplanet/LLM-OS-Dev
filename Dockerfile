FROM phidata/python:3.12

ARG USER=app
ARG APP_DIR=/app
ARG DOCKER_UID=61000
ENV APP_DIR=${APP_DIR}

# Create user and home directory
RUN groupadd -g ${DOCKER_UID} ${USER} \
  && useradd -g ${DOCKER_UID} -u ${DOCKER_UID} -ms /bin/bash -d ${APP_DIR} ${USER}

WORKDIR ${APP_DIR}

# Copy requirements.txt
COPY requirements/prod.txt ./requirements.txt

# Install requirements
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync requirements.txt --system

# Copy project files
COPY . .

RUN mkdir -p ${APP_DIR}/.composio

# Set permissions for the /app directory
RUN chown -R ${USER}:${USER} ${APP_DIR}

# Switch to non-root user
USER ${USER}

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["chill"]
