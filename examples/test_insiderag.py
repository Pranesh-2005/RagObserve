"""Drives the real InsideRag pipeline (F:/InsideRag/app.py) end-to-end with
sample text and checks that RAGObserve captured chunking, embedding, retrieval,
BM25+vector fusion and reranking events."""
import sys

sys.path.insert(0, "F:/InsideRag")
import app  # noqa: E402  (loads the sentence-transformer model)

import ragobserve  # noqa: E402

TEXT = (
    "The master services agreement between Acme Corporation and the client becomes "
    "effective on the first of January. Either party may terminate the agreement by "
    "providing ninety days written notice to the other party. All invoices issued under "
    "this agreement are payable within thirty days of receipt and late payments accrue "
    "monthly interest. The total liability of either party under this agreement shall "
    "not exceed the fees paid during the preceding twelve months. Confidential "
    "information includes all technical business and financial information disclosed by "
    "either party. The confidentiality obligations described in this section survive "
    "termination of the agreement for a period of five years. All employee laptops must "
    "use full disk encryption together with automatic screen locking policies. Access "
    "to production systems requires hardware key based two factor authentication for "
    "all engineers and administrators."
)

QUESTION = "how many days notice to terminate the agreement"

print("1/4 chunking...")
df, chunks = app.run_chunking(TEXT, 25)
print(f"    {len(chunks)} chunks")

print("2/4 embeddings + FAISS index...")
info, vecs, index = app.build_embeddings(chunks)

print("3/4 retrieval tab...")
df2, fig = app.retrieve(QUESTION, chunks, vecs, index, 3)
print(df2[["Chunk ID", "Distance"]].to_string(index=False))

print("4/4 hybrid answer (bm25 + vector -> rrf -> cosine rerank)...")
answer = app.generate_answer(QUESTION, chunks, vecs, index, 2)
print(answer.splitlines()[2][:120])

ragobserve.flush()
print("\nDone. Events written to F:/rag obs/ragobserve.db under project 'insiderag'.")
