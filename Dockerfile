FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts

# Install CPU-only torch FIRST, from PyTorch's CPU index. bge-m3 (sentence-transformers) pulls
# torch transitively, and the default PyPI wheel is the CUDA build: ~2 GB of nvidia_* wheels that
# unpack to ~5 GB and can never execute here — no GPU is passed into these containers. Installing
# the CPU wheel up front makes the following `pip install .` see torch as already satisfied, so it
# never reaches for the CUDA build. Drop this line if you ever add GPU passthrough.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

# The huggingface cache dir must exist AND be owned by appuser in the image: Docker initializes a
# named volume from the image path it covers, so a missing/root-owned dir yields a root-owned
# volume and the bge-m3 download dies with PermissionError under USER appuser.
RUN mkdir -p /app/data \
    && useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app/data /home/appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn multilingual_rag.api.app:app --host 0.0.0.0 --port 8000"]
