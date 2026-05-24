FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git pandoc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pydantic \
    pandas \
    openpyxl \
    requests \
    python-docx \
    pypdf \
    pypandoc \
    pypandoc_binary \
    llama-index \
    llama-index-embeddings-openai \
    llama-index-llms-openai \
    llama-index-llms-openai-like \
    python-telegram-bot
