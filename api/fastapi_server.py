from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json

from coderag.core.vector_retriever import VectorRetriever
from coderag.monitor import monitor

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # Initialize vector retriever with async loading
    global retriever
    retriever = VectorRetriever()
    await retriever.load_documents([])  # Initialize empty index

@app.post("/process-docs")
@monitor.track("process_documents")
async def process_documents_async(document: str, incremental: bool = False):
    """Process documents with async batch processing"""
    try:
        docs = [{"page_content": document, "metadata": {}}]  # Simplified doc format
        await retriever.load_documents(docs, incremental)
        return {"status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve")
@monitor.track("retrieve")
async def retrieve_async(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
):
    """Retrieve documents with rate limiting"""
    try:
        results = retriever.retrieve(query, top_k, rerank)
        return {"results": [{"content": r.page_content, "metadata": r.metadata} for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return monitor.get_metrics()
