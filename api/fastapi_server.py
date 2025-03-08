from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional
import uvicorn
from pydantic import BaseModel
from coderag.search import answer_with_rag
from coderag.monitor import track_performance

app = FastAPI(title="CodeRAG API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

@app.post("/query")
@limiter.limit("10/minute")
async def rag_query(request: QueryRequest):
    """Main RAG endpoint with rate limiting"""
    try:
        answer, docs = answer_with_rag(
            question=request.question,
            top_k=request.top_k
        )
        return {
            "answer": answer,
            "sources": [doc.metadata["source"] for doc in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
