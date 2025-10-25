# csibilevente14/mlflow-hello:2025-10-25-1620

FROM python:3.11-slim

WORKDIR /app

COPY track.py .

RUN pip install --no-cache-dir mlflow

CMD ["python", "track.py"]