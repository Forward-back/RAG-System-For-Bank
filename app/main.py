import os
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from app.schemas import QueryRequest, QueryResponse
from typing import Optional
from src.rag_pipeline import RAGPipeline
import logging
import time
from app.logger import get_logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure root logger so all src.* module loggers inherit the handler
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "recursive")
TOP_K = int(os.getenv("TOP_K", 5))

ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
QUERY_EXPANSION_MODE = os.getenv("QUERY_EXPANSION_MODE", "template")
ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
CACHE_PATH = os.getenv("CACHE_PATH", "./cache/query_cache.json")
ENABLE_NUMERIC_GUARD = os.getenv("ENABLE_NUMERIC_GUARD", "true").lower() == "true"
ENABLE_NLI_VERIFIER = os.getenv("ENABLE_NLI_VERIFIER", "true").lower() == "true"
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
TRACE_LOGS_DIR = os.getenv("TRACE_LOGS_DIR", "./logs")

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
WEB_SEARCH_QUALITY_THRESHOLD = float(os.getenv("WEB_SEARCH_QUALITY_THRESHOLD", "0.0"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "10.0"))

REBUILD_INDEX_ON_STARTUP = (
    os.getenv("REBUILD_INDEX_ON_STARTUP", "false").lower() == "true"
)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_LOCAL_FILES_ONLY = os.getenv("RERANKER_LOCAL_FILES_ONLY", "false").lower() == "true"
PDF_STRATEGY = os.getenv("PDF_STRATEGY", "auto_detect")

# Rate limiter: 30 requests per minute by client IP
limiter = Limiter(key_func=get_remote_address)

# API key authentication
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        return  # auth disabled when not configured
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


app = FastAPI(
    title="Production RAG API",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = get_logger("rap-api")
# Global pipeline instance
pipeline: Optional[RAGPipeline] = None

@app.on_event("startup")
def startup_event():

    global pipeline
    logger.info("Starting RAG API")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logger.warning("Data directory created dynamically.")

    try:
        pipeline=RAGPipeline(
            data_paths=[DATA_PATH],
            persist_dir=CHROMA_DIR,
            chunking_mode=CHUNKING_MODE,
            enable_rerank=ENABLE_RERANK,
            enable_compression=ENABLE_COMPRESSION,
            query_expansion_mode=QUERY_EXPANSION_MODE,
            enable_query_cache=ENABLE_QUERY_CACHE,
            cache_path=CACHE_PATH,
            enable_numeric_guard=ENABLE_NUMERIC_GUARD,
            enable_nli_verifier=ENABLE_NLI_VERIFIER,
            enable_tracing=ENABLE_TRACING,
            trace_logs_dir=TRACE_LOGS_DIR,
            top_k=TOP_K,
            reranker_model=RERANKER_MODEL,
            reranker_local_files_only=RERANKER_LOCAL_FILES_ONLY,
            pdf_strategy=PDF_STRATEGY,
            enable_web_search=ENABLE_WEB_SEARCH,
            web_search_quality_threshold=WEB_SEARCH_QUALITY_THRESHOLD,
            web_search_max_results=WEB_SEARCH_MAX_RESULTS,
            web_search_timeout=WEB_SEARCH_TIMEOUT,
            verbose=False
        )


        start=time.time()
        pipeline.build_index(rebuild=REBUILD_INDEX_ON_STARTUP)
        logger.info(f"Index ready in {time.time() - start:.2f}s")
        
        start=time.time()
        pipeline.load_models()
        logger.info(f"Models loaded in {time.time() - start:.2f}s")
        
        logger.info("RAG API startup complete with Groq backend")
        
    except Exception as e:
        logger.info("Startup failed")
        pipeline=None
        raise RuntimeError(f"Startup failed: {e}")

        
@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
def query_rag(request: Request, query_request: QueryRequest, _: None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    logger.info(f"Query received: {query_request.query}")

    start = time.time()

    try:
        answer = pipeline.run(
            query_request.query,
            enable_web_search=query_request.enable_web_search,
        )
        duration = time.time() - start

        logger.info(f"Query complete in {duration:.2f}s")

        return QueryResponse(answer=answer)

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/rebuild-index")
@limiter.limit("2/minute")
def rebuild_index(request: Request, _: None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    logger.warning("Index rebuild triggered")

    start = time.time()

    try:
        pipeline.rebuild_index()
        logger.warning(f"Index rebuild in {time.time() - start:.2f}s")
        return {"status": "index rebuild"}

    except Exception as e:
        logger.exception("Index rebuild failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        