# syntax=docker/dockerfile:1.7

FROM python:3.14-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.5,<3" \
    && python -m pip install ".[data,shawtynet]"


FROM python:3.14-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="OPHANIM" \
      org.opencontainers.image.description="Experimental ionospheric TEC forecasting and disturbance detection" \
      org.opencontainers.image.version="0.1.0"

ARG APP_UID=1000
ARG APP_GID=1000

ENV HOME=/home/ophanim \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN groupadd --gid "${APP_GID}" ophanim \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home \
        --shell /usr/sbin/nologin \
        ophanim \
    && install -d \
        -o "${APP_UID}" \
        -g "${APP_GID}" \
        /var/lib/ophanim \
        /var/lib/ophanim-zarr

COPY --from=builder /opt/venv /opt/venv

WORKDIR /var/lib/ophanim
USER ${APP_UID}:${APP_GID}

EXPOSE 8765
STOPSIGNAL SIGTERM

ENTRYPOINT ["ophanim-desktop"]
CMD ["--bind-address", "0.0.0.0", "--port", "8765", \
     "--data-dir", "/var/lib/ophanim", "--no-browser"]
