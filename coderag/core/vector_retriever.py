from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional

from .reranker import ColbertReranker
import hashlib
import json
from datetime import datetime

class VectorRetriever:
    def __init__(self, 
                 model_name="thenlper/gte-small",
                 reranker_model="colbert-ir/colbertv2.0",
                 redis_host="localhost",
                 redis_port=6379):
        # Version tracking system
        self.index_version = 1
        self.document_versions = {}  # Track source -> version mapping
        
        # Embedding model and vector store
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Caching and reranking components
        self.cache = CacheManager()
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
    
    def _get_versioned_hash(self, document: Document) -> tuple:
        """Return (base_hash, version) tuple using metadata version if exists"""
        base_hash = self._get_doc_hash(document)
        return (base_hash, document.metadata.get('version', 1))

    def load_documents(self, documents: List[Document], incremental: bool = False):
        """Process and index documents with version tracking and incremental updates"""
        processed_docs = []
        
        # Update document versions
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            self.document_versions[source] = self.document_versions.get(source, 0) + 1
            
            doc_hash = self._get_doc_hash(doc)
            cached_emb = self.cache.get_cached_embeddings(doc_hash)
            
            if cached_emb is None:
                # Track document versions before splitting
                base_hash, version = self._get_versioned_hash(doc)
                for split_doc in self.text_splitter.split_documents([doc]):
                    split_doc.metadata.update({
                        'base_hash': base_hash,
                        'version': version,
                        'last_updated': datetime.now().isoformat()
                    })
                    processed_docs.append(split_doc)
                # Cache embeddings with version info
                self.cache.cache_embeddings(base_hash, {
                    'embedding': self.embedding_model.embed_documents([doc.page_content]),
                    'version': version
                })
            else:
                # Create document with cached embeddings
                fake_doc = Document(page_content=doc.page_content, metadata=doc.metadata)
                fake_doc.embedding = cached_emb
                processed_docs.append(fake_doc)

        if incremental and self.vector_store:
            # Get current versions from index
            existing_hashes = {doc.metadata['base_hash']: doc.metadata['version'] 
                             for doc in self.vector_store.docstore._dict.values()
                             if 'base_hash' in doc.metadata}
            
            # Filter documents to only keep newer versions
            new_docs = [
                doc for doc in processed_docs
                if doc.metadata['base_hash'] not in existing_hashes 
                or doc.metadata['version'] > existing_hashes[doc.metadata['base_hash']]
            ]
            
            if new_docs:
                # Remove old versions before adding new ones
                ids_to_remove = [
                    idx for idx, doc in self.vector_store.docstore._dict.items()
                    if doc.metadata['base_hash'] in [d.metadata['base_hash'] for d in new_docs]
                ]
                
                self.vector_store.delete(ids_to_remove)
                self.vector_store.add_documents(new_docs)
        else:
            self.vector_store = FAISS.from_documents(
                processed_docs, 
                self.embedding_model,
                distance_strategy="COSINE"
            )
            self.index_version += 1  # Increment version on full rebuild

    def save_snapshot(self, snapshot_name: str = None):
        """Save versioned snapshot of the index with metadata"""
        if not self.vector_store:
            raise ValueError("No vector store initialized")
            
        snapshot_name = snapshot_name or f"v{self.index_version}"
        path = f"faiss_snapshots/{snapshot_name}"
        
        # Save FAISS index
        self.vector_store.save_local(path)
        
        # Save version metadata
        metadata = {
            "version": self.index_version,
            "timestamp": datetime.now().isoformat(),
            "document_versions": self.document_versions,
            "document_count": self.vector_store.index.ntotal
        }
        
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)
            
        return path

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
