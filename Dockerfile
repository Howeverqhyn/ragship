FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY demo ./demo

RUN pip install --no-cache-dir .

USER appuser

EXPOSE 8091

CMD ["python", "-m", "ragship", "serve"]
