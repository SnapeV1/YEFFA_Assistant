import logging
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict

# Logger setup
logger = logging.getLogger("yeffa.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Use AI-first matcher (semantic + ML tags)
try:
    from ai_matcher import respond, load_faq
    logger.info("Using matcher implementation: ai_matcher")
except Exception:
    from matcher import respond, load_faq
    logger.info("Using matcher implementation: matcher (fallback)")


app = FastAPI(title="YEFFA FAQ Matcher", version="1.0.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request to help debug host-to-VM connectivity issues."""
    start = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "Incoming %s %s from %s query=%s",
        request.method,
        request.url.path,
        client_host,
        request.url.query or "-",
    )
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Completed %s %s status=%s duration=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/respond")
def respond_options():
    """
    Explicitly handle OPTIONS so FastAPI/Uvicorn do not return 405.
    This enables proper CORS preflight for Angular.
    """
    return {"status": "ok"}


class Query(BaseModel):
    message: str
    include_debug: Optional[bool] = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/respond")
def post_respond(q: Query) -> Dict[str, Any]:
    logger.info("POST /respond include_debug=%s", q.include_debug)
    faq_path = os.path.join(os.path.dirname(__file__), "faq.json")
    logger.info("Loading FAQ file %s", faq_path)

    res = respond(q.message, load_faq(faq_path))

    payload = {
        "id": res.get("id"),
        "answer": res.get("answer"),
        "confidence": res.get("confidence"),
        "confidenceScore": res.get("confidenceScore"),
        "intent": res.get("intent"),
        "matchedTags": res.get("matchedTags", []),
    }

    if q.include_debug:
        payload["debug"] = res.get("debug")

    logger.info(
        "Matched id=%s intent=%s confidence=%s score=%s",
        payload.get("id"),
        payload.get("intent"),
        payload.get("confidence"),
        payload.get("confidenceScore"),
    )

    return payload


@app.get("/respond")
def get_respond(message: str, include_debug: bool = False) -> Dict[str, Any]:
    logger.info("GET /respond include_debug=%s", include_debug)
    faq_path = os.path.join(os.path.dirname(__file__), "faq.json")
    logger.info("Loading FAQ file %s", faq_path)

    res = respond(message, load_faq(faq_path))

    payload = {
        "id": res.get("id"),
        "answer": res.get("answer"),
        "confidence": res.get("confidence"),
        "confidenceScore": res.get("confidenceScore"),
        "intent": res.get("intent"),
        "matchedTags": res.get("matchedTags", []),
    }

    if include_debug:
        payload["debug"] = res.get("debug")

    logger.info(
        "Matched id=%s intent=%s confidence=%s score=%s",
        payload.get("id"),
        payload.get("intent"),
        payload.get("confidence"),
        payload.get("confidenceScore"),
    )

    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)

