FROM python:3.11-slim

LABEL maintainer="Gaurav Srivastava <gks@vt.edu>"
LABEL description="effGen — Agentic AI framework for Small Language Models"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY . .
RUN pip install --no-cache-dir -e .

# Verify install
RUN python -c "from effgen import Agent, load_model; print('effGen installed')"

# Default: start the API server
EXPOSE 8000
ENV EFFGEN_RATE_LIMIT=60

ENTRYPOINT ["effgen"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
