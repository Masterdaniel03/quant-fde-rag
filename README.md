# Quantitative Financial Filing & Market Data RAG Engine

A financial research tool that combines SEC 10-K report text with live stock market data using AI.

## What It Does
- **Document Search**: Indexes company SEC filings in an in-memory Qdrant vector database.
- **Market Metrics**: Calculates live annualized volatility, Sharpe ratio, and drawdowns via Yahoo Finance and NumPy.
- **AI Synthesis**: Uses Google Gemini to generate structured analysis grounded in filing text.
- **Fact Verification**: Automatically checks generated summaries to ensure reported numbers match calculated figures.

## Tech Stack
- Python, FastAPI
- Qdrant (Vector Database)
- Google Gemini API
- NumPy, yfinance

## Quickstart
```bash
# Set your Gemini API key
$env:GEMINI_API_KEY="your_api_key_here"

# Run the terminal pipeline
python llm_client.py

# Or launch the interactive web dashboard
uvicorn app:app --reload --port 8080 
