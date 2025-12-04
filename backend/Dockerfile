FROM python:3.10-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/backend/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
# COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
