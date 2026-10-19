FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir numpy pandas yfinance qdrant-client rank-bm25 langgraph fastapi uvicorn pydantic
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]