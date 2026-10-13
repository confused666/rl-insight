# RL-Insight observability image.
# Bundles the rl-insight CLI with Prometheus, Tempo, and Grafana, installed at
# build time by `rl-insight server install` (versions pinned in the package).
# Target: rl-insight:latest.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/verl-project/rl-insight"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# tini reaps orphaned service processes; ca-certificates for release downloads.
# curl/wget/zip/unzip/vim/less/procps/net-tools/iputils-ping are common
# debugging tools bundled for convenience. "ll" is a shell alias (not a
# package), so append it to /root/.bashrc for interactive shells.
# The container runs as root for easier in-container debugging; service
# state and manifests therefore live under /root/.rl-insight.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates tini curl wget zip unzip vim less procps net-tools iputils-ping \
    && rm -rf /var/lib/apt/lists/* \
    && echo "alias ll='ls -alF'" >> /root/.bashrc

WORKDIR /opt/rl-insight

# Install the Python package before the service binaries so code changes do
# not invalidate the expensive download layers. PIP_INDEX_URL defaults to
# PyPI; users in China can pass a mirror for faster downloads:
# docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
COPY pyproject.toml README.md LICENSE ./
COPY rl_insight/ ./rl_insight/
COPY recipe/ ./recipe/
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install .

# Bundle Prometheus, Tempo, and Grafana. Drop the download cache, which is
# never needed at container startup.
RUN rl-insight server install \
    && rm -rf /root/.rl-insight/services/_downloads \
    && mkdir -p /root/.rl-insight/data

COPY docker/entrypoint.py docker/healthcheck.py /opt/rl-insight/docker/

VOLUME ["/root/.rl-insight/data"]

# 18080 rl-insight API, 3000 Grafana, 9090 Prometheus, 3200 Tempo, 4318 OTLP/HTTP.
EXPOSE 18080 3000 9090 3200 4318

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD ["python", "/opt/rl-insight/docker/healthcheck.py"]

# tini forwards Docker's stop signal; the entrypoint maps SIGTERM onto the
# CLI's Ctrl+C path, which stops all four service processes gracefully.
ENTRYPOINT ["/usr/bin/tini", "-e", "130", "--", "python", "/opt/rl-insight/docker/entrypoint.py"]
CMD ["server", "start", "--attach-logs"]
