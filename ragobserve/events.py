"""Universal RAG event model.

Every RAG system, regardless of framework, passes through the same lifecycle:

    ingestion -> chunking -> embedding -> indexing -> retrieval -> fusion
    -> reranking -> context_assembly -> prompt -> generation -> grounding

A single ``RagEvent`` schema covers every stage; stage-specific payloads live
in ``attributes`` using the documented keys below. Field names follow the
OpenTelemetry GenAI semantic conventions where one exists so a future OTLP
bridge is cheap.

Stage attribute conventions
---------------------------
ingestion:        source, file_size, parse_errors
chunking:         chunks=[{id?, text, source?, token_count?, metadata?}], strategy, chunk_size, overlap
embedding:        model, dimensions, input_count, cost
indexing:         index, vector_db, count
retrieval:        query, top_k, retriever, results=[{id?, text?, score, rank?, source?, metadata?}]
fusion:           strategy, inputs={name: results[]}, results=[...]
reranking:        model, before=[results], after=[results]
context_assembly: system_prompt, query, chunks=[...], final_prompt, token_count, context_window
generation:       model, prompt, response, input_tokens, output_tokens, cost
grounding:        claims=[{sentence, chunk_id?, supported}]
"""
from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    RERANKING = "reranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    PROMPT = "prompt"
    GENERATION = "generation"
    GROUNDING = "grounding"
    AGENT = "agent"
    TOOL = "tool"
    OTHER = "other"


STAGES = [s.value for s in Stage]


def new_id() -> str:
    return uuid.uuid4().hex


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) so we avoid a tokenizer dependency."""
    return max(1, len(text) // 4)


class RagEvent(BaseModel):
    event_id: str = Field(default_factory=new_id)
    trace_id: str = Field(default_factory=new_id)
    span_id: str = Field(default_factory=new_id)
    parent_span_id: Optional[str] = None
    project: str = "default"
    stage: str = Stage.OTHER.value
    name: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok | error
    attributes: Dict[str, Any] = Field(default_factory=dict)

    def finish(self, status: str = "ok") -> "RagEvent":
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status
        return self


class Chunk(BaseModel):
    chunk_id: Optional[str] = None
    text: str = ""
    source: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def hashed(self) -> str:
        return self.chunk_id or content_hash(self.text)


def normalize_result(item: Any) -> Dict[str, Any]:
    """Coerce a retrieval result (str | dict | Chunk-like) into the wire shape."""
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, Chunk):
        item = item.model_dump()
    if isinstance(item, dict):
        out = dict(item)
        if "id" not in out and "chunk_id" in out:
            out["id"] = out.pop("chunk_id")
        return out
    # objects with .text / .page_content (LangChain / LlamaIndex docs)
    text = getattr(item, "text", None) or getattr(item, "page_content", None)
    if text is not None:
        return {
            "text": text,
            "metadata": dict(getattr(item, "metadata", {}) or {}),
        }
    return {"text": str(item)}
