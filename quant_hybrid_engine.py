import hashlib
import numpy as np
import yfinance as yf
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