FROM python:3.10-slim

# Install C++ compiler for llama-cpp
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir huggingface_hub llama-cpp-python gunicorn flask requests

COPY . .

# Railway uses the PORT env var
ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]