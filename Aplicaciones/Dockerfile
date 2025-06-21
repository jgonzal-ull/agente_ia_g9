# Usa una imagen base de PyTorch con soporte para CUDA
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Instala PyTorch con soporte para CUDA y los Transformers de Hugging Face
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir transformers

# Copia el código de la aplicación dentro del contenedor (opcional)
# COPY requirements.txt /app/
# RUN pip install -r /app/requirements.txt

# Hugging Face
RUN git config --global credential.helper store

# Specific library
RUN pip install pydantic
RUN pip install pandas
RUN pip install python-docx
RUN pip install scikit-learn
RUN pip install sentence-transformers
RUN pip install shap
RUN pip install matplotlib
RUN pip install pyarrow
RUN pip install einops
RUN pip install datasets
RUN pip install 'accelerate>=0.26.0'
RUN pip install evaluate
RUN pip install openpyxl
RUN pip install pypdf
RUN pip install pypandoc
RUN pip install pypandoc_binary
RUN pip install llama-index
RUN pip install llama-index-embeddings-openai
RUN pip install llama-index-llms-openai-like
RUN pip install python-telegram-bot

