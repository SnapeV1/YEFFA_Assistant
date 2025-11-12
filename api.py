import logging
import os
from fastapi import FastAPI
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
    # Fallback to previous matcher if AI module import fails
    from matcher import respond, load_faq
    logger.info("Using matcher implementation: matcher (fallback)")


app = FastAPI(title="YEFFA FAQ Matcher", version="1.0.0")

# CORS: allow local frontend (Angular at :4200) to call this API
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str
    include_debug: Optional[bool] = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/respond")
def post_respond(q: Query) -> Dict[str, Any]:
    logger.info("POST /respond include_debug=%s", q.include_debug)
    faq_path = os.path.join(os.path.dirname(__file__), 'faq.json')
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
        payload.get("id"), payload.get("intent"), payload.get("confidence"), payload.get("confidenceScore")
    )
    return payload


@app.get("/respond")
def get_respond(message: str, include_debug: bool = False) -> Dict[str, Any]:
    logger.info("GET /respond include_debug=%s", include_debug)
    faq_path = os.path.join(os.path.dirname(__file__), 'faq.json')
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
        payload.get("id"), payload.get("intent"), payload.get("confidence"), payload.get("confidenceScore")
    )
    return payload
if __name__ == "__main__":
    # Allow running via: python api.py
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=5001, reload=True)
