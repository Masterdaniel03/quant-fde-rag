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