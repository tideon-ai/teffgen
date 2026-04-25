FROM python:3.11-slim

LABEL maintainer="Gaurav Srivastava <gks@vt.edu>"
LABEL description="tideon.ai — Agentic AI framework for Small Language Models"
LABEL version="0.2.0"

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
RUN python -c "import teffgen; print(f'tideon.ai {teffgen.__version__} installed')"
RUN python -c "from teffgen.guardrails import GuardrailChain; print('Guardrails OK')"
RUN python -c "from teffgen.rag import DocumentIngester; print('RAG OK')"
RUN python -c "from teffgen.eval import AgentEvaluator; print('Eval OK')"
RUN python -c "from teffgen.domains import TechDomain; print('Domains OK')"

# Default: start the API server
EXPOSE 8000
ENV TEFFGEN_RATE_LIMIT=60

ENTRYPOINT ["teffgen"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
