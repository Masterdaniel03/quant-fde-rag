import hashlib
import re

import numpy as np
from curl_cffi.requests.exceptions import RequestException
import yfinance as yf
from yfinance.exceptions import (
    YFException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
)
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QuantHybridEngine:
    def __init__(self, embedding_dim: int = 128):
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "sec_10k_filings"
        self.embedding_dim = embedding_dim
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )
        self.documents = []
        self.bm25 = None

    def _deterministic_embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.embedding_dim
            sign = 1.0 if (h // self.embedding_dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def ingest_unstructured_filings(self, filing_chunks: list[dict]):
        points = []
        tokenized_corpus = []
        for idx, item in enumerate(filing_chunks):
            self.documents.append(item)
            tokenized_corpus.append(item["text"].lower().split())
            vector = self._deterministic_embed(item["text"])
            points.append(PointStruct(id=idx, vector=vector, payload=item))
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_retrieve_rrf(self, query: str, top_k: int = 2, k_rrf: int = 60) -> list[dict]:
        if not self.documents or self.bm25 is None:
            return []
        query_tokens = query.lower().split()
        query_vec = self._deterministic_embed(query)
        vector_hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=len(self.documents),
        ).points

        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        n_docs = len(self.documents)
        rrf_scores = np.zeros(n_docs, dtype=np.float64)

        for rank, hit in enumerate(vector_hits):
            rrf_scores[int(hit.id)] += 1.0 / (k_rrf + rank + 1)

        for rank, doc_id in enumerate(bm25_ranks):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["rrf_score"] = round(float(rrf_scores[idx]), 5)
            results.append(doc)
        return results

    def compute_structured_market_metrics(
        self, ticker: str, period: str = "6mo"
    ) -> dict[str, float | str | None]:
        normalized_ticker = ticker.strip().upper()
        fallback_reason: str | None = None

        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", normalized_ticker):
            fallback_reason = "invalid_ticker"
            df = None
        else:
            try:
                df = yf.Ticker(normalized_ticker).history(period=period)
            except YFRateLimitError:
                df = None
                fallback_reason = "rate_limited"
            except (YFTickerMissingError, YFPricesMissingError):
                df = None
                fallback_reason = "ticker_unavailable"
            except YFException:
                df = None
                fallback_reason = "market_data_unavailable"
            except (RequestException, TimeoutError, OSError):
                df = None
                fallback_reason = "market_data_unavailable"

        market_data_source = "yfinance"
        if df is None or df.empty or "Close" not in df:
            prices = None
        else:
            prices = df["Close"].dropna().to_numpy(dtype=np.float64)

        if prices is None or len(prices) < 5 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            if fallback_reason is None:
                fallback_reason = "insufficient_market_data"
            market_data_source = "deterministic_fallback"
            returns = np.random.RandomState(42).normal(0.0012, 0.021, 126)
            prices = 100.0 * np.exp(np.cumsum(returns))
        else:
            returns = np.diff(prices) / prices[:-1]

        ann_volatility = float(np.std(returns) * np.sqrt(252))
        ann_return = float(np.mean(returns) * 252)
        sharpe_ratio = float((ann_return - 0.045) / (ann_volatility + 1e-9))

        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

        return {
            "ticker": normalized_ticker,
            "annualized_volatility_pct": round(ann_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "latest_price": round(float(prices[-1]), 2),
            "market_data_source": market_data_source,
            "fallback_reason": fallback_reason,
        }