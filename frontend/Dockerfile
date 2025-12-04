FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir streamlit requests

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

