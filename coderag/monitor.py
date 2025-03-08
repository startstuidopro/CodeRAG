import time
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter, Gauge

# Prometheus metrics
REQUEST_LATENCY = Summary('rag_request_latency_seconds', 'RAG request processing latency')
REQUEST_COUNT = Counter('rag_request_count', 'Total RAG requests')
ERROR_COUNT = Counter('rag_error_count', 'Total RAG errors')
CACHE_HIT_RATE = Gauge('rag_cache_hit_rate', 'Cache hit rate percentage')

class QueryAnalytics:
    def __init__(self):
        self.queries = []
        self.metrics = {
            'latency': [],
            'cache_hits': 0,
            'total_requests': 0
        }
    
    def track_query(self, query: str, sources: List[str], latency: float):
        self.queries.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "sources": sources,
            "latency": latency
        })
        self.metrics['total_requests'] += 1
        self.metrics['latency'].append(latency)
        
    def track_cache_hit(self):
        self.metrics['cache_hits'] += 1
        CACHE_HIT_RATE.set(self.cache_hit_percentage)
    
    @property
    def cache_hit_percentage(self) -> float:
        if self.metrics['total_requests'] == 0:
            return 0.0
        return (self.metrics['cache_hits'] / self.metrics['total_requests']) * 100
    
    def get_latency_stats(self) -> Dict[str, float]:
        series = pd.Series(self.metrics['latency'])
        return {
            "mean": series.mean(),
            "p95": series.quantile(0.95),
            "max": series.max()
        }

def start_monitoring(port=9100):
    """Start metrics server and background monitoring"""
    start_http_server(port)
    print(f"Monitoring server started on port {port}")

# Global analytics instance
analytics = QueryAnalytics()

def track_performance(func):
    """Decorator to track RAG method performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            
            # Track metrics
            REQUEST_LATENCY.observe(latency)
            REQUEST_COUNT.inc()
            
            if 'question' in kwargs and 'docs' in result:
                analytics.track_query(
                    query=kwargs['question'],
                    sources=[doc.metadata.get('source', '') for doc in result['docs']],
                    latency=latency
                )
            
            return result
        except Exception as e:
            ERROR_COUNT.inc()
            raise
    
    return wrapper
