import os
import time
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

import ragobserve

load_dotenv()

ragobserve.init(project="hybrid-rag")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

DOCUMENTS = [
    "Retrieval Augmented Generation combines retrieval with LLM generation.",
    "BM25 is a lexical search algorithm based on term frequency and inverse document frequency.",
    "Vector search uses embeddings to find semantically similar documents.",
    "Reranking improves retrieval quality by rescoring candidate documents.",
    "Groq provides fast inference for open-source large language models.",
    "Hybrid search combines lexical and semantic retrieval methods."
]

tokenized_docs = [doc.lower().split() for doc in DOCUMENTS]
bm25 = BM25Okapi(tokenized_docs)

embeddings = EMBED_MODEL.encode(
    DOCUMENTS,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


def retrieve(query, top_k=5):

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    query_emb = EMBED_MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, ids = index.search(
        query_emb,
        top_k
    )

    vector_results = [
        {"id": int(doc_id), "text": DOCUMENTS[doc_id], "score": float(scores[0][rank])}
        for rank, doc_id in enumerate(ids[0])
    ]
    bm25_results = [
        {"id": i, "text": DOCUMENTS[i], "score": float(s)}
        for i, s in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
    ]
    ragobserve.log_retrieval(query, vector_results, top_k=top_k, retriever="faiss-IndexFlatIP")

    hybrid_scores = {}

    for idx, score in enumerate(bm25_scores):
        hybrid_scores[idx] = score

    for rank, doc_id in enumerate(ids[0]):
        hybrid_scores[doc_id] = (
            hybrid_scores.get(doc_id, 0)
            + float(scores[0][rank]) * 5
        )

    candidates = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    fused_results = [
        {"id": int(i), "text": DOCUMENTS[i], "score": float(s)}
        for i, s in candidates
    ]
    ragobserve.log_fusion(
        fused_results,
        inputs={"bm25": bm25_results, "vector": vector_results},
        strategy="weighted sum (bm25 + vector*5)",
    )

    candidate_docs = [
        DOCUMENTS[i]
        for i, _ in candidates
    ]

    pairs = [
        [query, doc]
        for doc in candidate_docs
    ]

    rerank_scores = RERANKER.predict(pairs)

    reranked = sorted(
        zip(candidate_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    ragobserve.log_rerank(
        before=[{"text": d, "score": float(s)} for d, s in zip(candidate_docs, rerank_scores)],
        after=[{"text": d, "score": float(s)} for d, s in reranked],
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    return reranked[:3]


def chat(query):

  with ragobserve.trace("query", query=query):

    retrieved = retrieve(query)

    context = "\n\n".join(
        [doc for doc, _ in retrieved]
    )

    prompt = f"""
Use only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    ragobserve.log_context(
        prompt,
        query=query,
        chunks=[{"text": doc, "score": float(s)} for doc, s in retrieved],
        context_window=8192,
    )

    model = "llama-3.1-8b-instant"
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )
    duration_ms = (time.time() - t0) * 1000.0

    answer = response.choices[0].message.content

    usage = getattr(response, "usage", None)
    ragobserve.log_generation(
        model=model,
        prompt=prompt,
        response=answer,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        duration_ms=duration_ms,
        provider="groq",
    )

    sources = "\n".join(
        [f"- {doc}" for doc, _ in retrieved]
    )

    return answer, sources


with gr.Blocks() as demo:

    gr.Markdown("# Hybrid RAG Demo")

    query = gr.Textbox(
        label="Question"
    )

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    sources = gr.Textbox(
        label="Retrieved Context",
        lines=10
    )

    btn = gr.Button("Ask")

    btn.click(
        chat,
        inputs=query,
        outputs=[answer, sources]
    )

demo.launch()