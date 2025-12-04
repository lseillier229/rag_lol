FROM python:3.10-slim

WORKDIR /app

COPY rag_core.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "rag_core.py"]

