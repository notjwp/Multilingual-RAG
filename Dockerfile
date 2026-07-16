FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir .

RUN mkdir -p /app/data \
    && useradd --create-home appuser \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn multilingual_rag.api.app:app --host 0.0.0.0 --port 8000"]
