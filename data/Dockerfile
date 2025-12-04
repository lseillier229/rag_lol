FROM python:3.10-slim

WORKDIR /app/data

COPY data /app/data

EXPOSE 9000

CMD ["python", "-m", "http.server", "9000", "--directory", "/app/data"]
