from ragatouille import RAGPretrainedModel
from typing import List, Dict
import numpy as np

class ColbertReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.reranker = RAGPretrainedModel.from_pretrained(model_name)
        
    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        cross_encoder_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Rerank documents using hybrid ColBERTv2 + cross-encoder scoring
        """
        # First-stage reranking with ColBERT
        colbert_results = self.reranker.rerank(query, documents, k=top_k*3)
        
        # Second-stage cross-encoder scoring
        scored_results = []
        for doc in colbert_results:
            score = self.reranker.cross_encoder_score(query, doc["content"])
            if score >= cross_encoder_threshold:
                scored_results.append({
                    "content": doc["content"],
                    "score": np.mean([doc["score"], score]),
                    "metadata": doc.get("metadata", {})
                })
                
        # Return top_k hybrid scored results
        return sorted(scored_results, key=lambda x: x["score"], reverse=True)[:top_k]
