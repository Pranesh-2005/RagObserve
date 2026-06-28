FROM python:3.11-slim

WORKDIR /app

# Install core + optional backends in one layer
COPY pyproject.toml README.md ./
COPY ragobserve/ ./ragobserve/

RUN pip install --no-cache-dir ".[postgres,files]"

# Data lives here — mount a volume to persist across restarts
VOLUME ["/data"]

ENV RAGOBSERVE_STORE=/data/ragobserve.db \
    RAGOBSERVE_HOST=0.0.0.0 \
    RAGOBSERVE_PORT=5601

EXPOSE 5601

# Single worker: in-process bus required for WebSocket live feed
CMD ["ragobserve", "ui", "--host", "0.0.0.0"]
