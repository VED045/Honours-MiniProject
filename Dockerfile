FROM python:3.12.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-torch.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r requirements-torch.txt
COPY backend backend
COPY ml ml
COPY scripts scripts
COPY docs docs
ENV PYTHONUNBUFFERED=1 FRAUD_HOME=/app/storage PORT=8000 TRAINING_ENABLED=false REQUIRE_API_KEY=true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"
CMD ["python","-m","backend.supervisor"]
