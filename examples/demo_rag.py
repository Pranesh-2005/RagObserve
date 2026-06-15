"""Simulates a small hybrid RAG pipeline using only the RAGObserve SDK so the
dashboard has data to explore. No API keys or vector DBs needed.

Run:  python examples/demo_rag.py
Then: ragobserve ui   (and open http://127.0.0.1:5601)
"""
import random
import time

import ragobserve

random.seed(7)

DOCS = {
    "contracts/msa_acme.md": [
        "The Master Services Agreement between Acme Corp and the Client takes effect on January 1, 2026.",
        "Either party may terminate this agreement with ninety (90) days written notice.",
        "All invoices are due within thirty (30) days of receipt. Late payments accrue 1.5% monthly interest.",
        "The liability of either party shall not exceed the fees paid in the preceding twelve months.",
    ],
    "contracts/nda_acme.md": [
        "Confidential Information includes all technical, business and financial information disclosed.",
        "The confidentiality obligations survive termination of this agreement for five (5) years.",
        "Either party may terminate this agreement with ninety (90) days written notice.",  # duplicate on purpose
    ],
    "policies/security.md": [
        "All employee laptops must use full-disk encryption and automatic screen locking.",
        "Access to production systems requires hardware-key two-factor authentication.",
        "This chunk is never retrieved by any query and shows up as a dead chunk.",
    ],
}

QUERIES = [
    ("What is the notice period for termination?", "terminat"),
    ("When are invoices due?", "invoice"),
    ("How long do confidentiality obligations last?", "confidential"),
    ("What is the liability cap?", "liability"),
    ("Do laptops need encryption?", "encrypt"),
]


def fake_score(text: str, needle: str) -> float:
    if "never retrieved" in text:
        return 0.01
    base = 0.92 if needle.lower() in text.lower() else random.uniform(0.2, 0.6)
    return round(min(0.99, base + random.uniform(-0.05, 0.05)), 4)


def main() -> None:
    ragobserve.init(project="contract-rag")

    # --- one-time ingestion: register the corpus chunks
    all_chunks = [
        {"text": text, "source": src}
        for src, chunks in DOCS.items()
        for text in chunks
    ]
    with ragobserve.trace("ingestion", source="local-markdown"):
        ragobserve.log_chunks(all_chunks, strategy="by-paragraph", chunk_size=256, overlap=0)
        ragobserve.log_embedding(model="text-embedding-3-small", input_count=len(all_chunks),
                               dimensions=1536, cost=0.0001, duration_ms=180.0)

    # --- simulated queries
    for question, needle in QUERIES:
        with ragobserve.trace("query", query=question):
            # hybrid retrieval: bm25 + vector, fused, then reranked
            scored = sorted(
                ({"text": t, "source": s, "score": fake_score(t, needle)}
                 for s, chunks in DOCS.items() for t in chunks),
                key=lambda c: c["score"], reverse=True,
            )
            bm25 = scored[:4]
            vector = sorted(scored[:6], key=lambda c: c["score"] * random.uniform(0.8, 1.2), reverse=True)[:4]
            time.sleep(0.02)
            ragobserve.log_retrieval(question, vector, retriever="faiss-demo",
                                   top_k=4, duration_ms=random.uniform(15, 45))
            fused = sorted({c["text"]: c for c in bm25 + vector}.values(),
                           key=lambda c: c["score"], reverse=True)[:5]
            ragobserve.log_fusion(fused, inputs={"bm25": bm25, "vector": vector}, strategy="rrf")

            reranked = sorted(fused, key=lambda c: c["score"] + random.uniform(-0.15, 0.15), reverse=True)
            time.sleep(0.01)
            ragobserve.log_rerank(fused, reranked, model="bge-reranker-demo",
                                duration_ms=random.uniform(30, 80))

            top = reranked[:3]
            context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in top)
            system = "Answer strictly from the provided context. Cite sources."
            final_prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}"
            ragobserve.log_context(final_prompt, system_prompt=system, query=question,
                                 chunks=top, context_window=8192)

            answer = f"Based on {top[0]['source']}: {top[0]['text']}"
            time.sleep(0.05)
            ragobserve.log_generation(model="gpt-4o-mini", prompt=final_prompt, response=answer,
                                    input_tokens=len(final_prompt) // 4, output_tokens=len(answer) // 4,
                                    cost=round(random.uniform(0.0004, 0.002), 5),
                                    duration_ms=random.uniform(300, 900))

            # ground truth: the chunks that actually contain the needle
            relevant = [ragobserve.Chunk(text=c["text"]).hashed()
                        for c in scored if needle.lower() in c["text"].lower()]
            ragobserve.log_ground_truth(relevant)

    ragobserve.flush()
    print("Demo data written to ./ragobserve.db — run `ragobserve ui` and open http://127.0.0.1:5601")


if __name__ == "__main__":
    main()
