from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional
from .caching import CacheManager
from .reranker import ColbertReranker
import hashlib
import json

class VectorRetriever:
    def __init__(self, 
                 model_name="thenlper/gte-small",
                 reranker_model="colbert-ir/colbertv2.0",
                 redis_host="localhost",
                 redis_port=6379):
        
        # Embedding model and vector store
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Caching and reranking components
        self.cache = CacheManager(host=redis_host, port=redis_port)
        self.reranker = ColbertReranker(reranker_model)
        self.vector_store = None

        # Document processing configuration
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.embedding_model.client.tokenizer,
            chunk_size=512,
            chunk_overlap=51,
            separators=["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", 
                      "\n___+\n", "\n\n", "\n", " ", ""]
        )

    def _get_doc_hash(self, document: Document) -> str:
        """Generate unique hash for document content and metadata"""
        content_str = f"{document.page_content}{json.dumps(document.metadata)}"
        return hashlib.sha256(content_str.encode()).hexdigest()

    def load_documents(self, documents: List[Document], incremental: bool = False):
        """Process and index documents with caching and incremental updates"""
        processed_docs = []
        for doc in documents:
            doc_hash = self._get_doc_hash(doc)
            cached_emb = self.cache.get_cached_embeddings(doc_hash)
            
            if cached_emb is None:
                split_docs = self.text_splitter.split_documents([doc])
                processed_docs.extend(split_docs)
                # Cache new embeddings
                self.cache.cache_embeddings(doc_hash, self.embedding_model.embed_documents([doc.page_content]))
            else:
                # Create document with cached embeddings
                fake_doc = Document(page_content=doc.page_content, metadata=doc.metadata)
                fake_doc.embedding = cached_emb
                processed_docs.append(fake_doc)

        if incremental and self.vector_store:
            self.vector_store.add_documents(processed_docs)
        else:
            self.vector_store = FAISS.from_documents(
                processed_docs, 
                self.embedding_model,
                distance_strategy="COSINE"
            )

    def retrieve(self, query: str, top_k: int = 5, use_reranking: bool = True) -> List[Document]:
        """Retrieve with caching and optional reranking"""
        # Check cache first
        cached_results = self.cache.get_cached_query_results(query)
        if cached_results:
            return cached_results

        # Initial retrieval
        if not self.vector_store:
            raise ValueError("Vector store not initialized - load documents first")
            
        docs = self.vector_store.similarity_search(query, k=top_k*3 if use_reranking else top_k)

        # Rerank if enabled
        if use_reranking:
            doc_contents = [doc.page_content for doc in docs]
            reranked = self.reranker.rerank_documents(query, doc_contents, top_k=top_k)
            docs = [Document(page_content=res["content"], metadata=res["metadata"]) for res in reranked]

        # Cache final results
        self.cache.cache_query_results(query, [dict(page_content=d.page_content, metadata=d.metadata) for d in docs])
        
        return docs[:top_k]
