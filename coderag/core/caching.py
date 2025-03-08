from typing import Dict, Any, Optional
import hashlib
import json
import numpy as np
from datetime import datetime

class CacheManager:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ttl: Dict[str, datetime] = {}

    def _generate_key(self, prefix: str, content: Any) -> str:
        """Create unique cache key using SHA-256 hash"""
        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        return f"{prefix}:{hashlib.sha256(content_str.encode()).hexdigest()}"

    def cache_embeddings(self, document_hash: str, embeddings: np.ndarray):
        """Store embeddings in memory"""
        key = self._generate_key("emb", document_hash)
        self.cache[key] = embeddings.tolist()
        self.ttl[key] = datetime.now()

    def get_cached_embeddings(self, document_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embeddings"""
        key = self._generate_key("emb", document_hash)
        return np.array(self.cache.get(key)) if key in self.cache else None

    def cache_query_results(self, query: str, results: list):
        """Cache query results in memory"""
        key = self._generate_key("query", query)
        self.cache[key] = results
        self.ttl[key] = datetime.now()

    def get_cached_query_results(self, query: str) -> Optional[list]:
        """Get cached query results"""
        key = self._generate_key("query", query)
        return self.cache.get(key)
