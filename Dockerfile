FROM python:3.11-slim

# -------------------------
# 1. System deps (for torch, lightgbm, numpy / scipy stack)
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 2. Workdir & env
# -------------------------
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# -------------------------
# 3. Install Python deps
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# 4. Copy project
# -------------------------
COPY . .

# -------------------------
# 5. Streamlit config
# -------------------------
EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
