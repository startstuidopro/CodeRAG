import redis
import json
from typing import Optional, Any
import hashlib
import numpy as np

class CacheManager:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl  # Time-to-live in seconds

    def _generate_key(self, prefix: str, content: Any) -> str:
        """Create unique cache key using SHA-256 hash"""
        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        return f"{prefix}:{hashlib.sha256(content_str.encode()).hexdigest()}"

    def cache_embeddings(self, document_hash: str, embeddings: np.ndarray):
        """Store embeddings with document hash as key"""
        key = self._generate_key("emb", document_hash)
        self.redis.set(key, json.dumps(embeddings.tolist()), ex=self.ttl)

    def get_cached_embeddings(self, document_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embeddings"""
        key = self._generate_key("emb", document_hash)
        cached = self.redis.get(key)
        return np.array(json.loads(cached)) if cached else None

    def cache_llm_response(self, query: str, context: str, response: str):
        """Cache LLM responses using query+context as key"""
        composite_key = {"query": query, "context": context}
        key = self._generate_key("llm", composite_key)
        self.redis.set(key, response, ex=self.ttl)

    def get_cached_response(self, query: str, context: str) -> Optional[str]:
        """Get cached LLM response"""
        composite_key = {"query": query, "context": context}
        key = self._generate_key("llm", composite_key)
        return self.redis.get(key)

    def cache_query_results(self, query: str, results: list, top_k: int = 5):
        """Cache retrieved documents for queries"""
        key = self._generate_key("query", query)
        self.redis.set(key, json.dumps(results[:top_k]), ex=self.ttl)

    def get_cached_query_results(self, query: str) -> Optional[list]:
        """Retrieve cached query results"""
        key = self._generate_key("query", query)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None
