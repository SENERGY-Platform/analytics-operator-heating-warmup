# The minor stays 3.10: Ray Client needs the Python of the Ray cluster (rayproject/ray:2.55.0 is py310).
ARG BASE=python:3.10.22-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d
# The uv of the Ray cluster image, which builds the training environment on the workers.
ARG UV=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754

FROM ${UV} AS uv

FROM ${BASE} AS revision
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY .git .git
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > /git_commit

FROM ${BASE}
LABEL org.opencontainers.image.source="https://github.com/SENERGY-Platform/analytics-operator-heating-warmup"
COPY --from=uv /uv /usr/local/bin/uv
# The environment lives outside the working directory, which Ray uploads to the workers for the training.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
# Ray pins the workers to this; the minor only, the cluster has another patch release (3.10.20).
ENV UV_PYTHON=3.10
WORKDIR /usr/src/app
COPY pyproject.toml uv.lock ./
# git only for the operator lib, which is installed from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && uv sync --frozen --no-dev \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=revision /git_commit git_commit
COPY main.py .
COPY algo algo
# Started by uv run, Ray runs the training tasks in the same locked environment on the workers.
CMD ["uv", "run", "--frozen", "main.py"]
