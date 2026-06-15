import os
import time
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import ragobserve
from ragobserve.adapters import instrument_embeddings

ragobserve.init(project="langchain-rag")

DOCS = [
    "Retrieval Augmented Generation combines retrieval with LLM generation.",
    "BM25 is a lexical search algorithm based on term frequency and inverse document frequency.",
    "Vector search uses embeddings to find semantically similar documents.",
    "Reranking improves retrieval quality by rescoring candidate documents.",
    "Groq provides fast inference for open-source large language models.",
    "Hybrid search combines lexical and semantic retrieval methods."
]

documents = [Document(page_content=d) for d in DOCS]

# instrument_embeddings -> embed_documents auto-logs an embedding event (ingest)
embeddings = instrument_embeddings(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

vectorstore = FAISS.from_documents(documents, embeddings)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# new-langchain idiomatic: retrievers are runnables, fusion + rerank done
# explicitly (EnsembleRetriever / ContextualCompressionRetriever live only in
# the legacy langchain_classic shim).
reranker = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


def reciprocal_rank_fusion(result_lists, k=60):
    scores, docmap = {}, {}
    for docs in result_lists:
        for rank, d in enumerate(docs):
            key = d.page_content
            docmap[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(scores, key=scores.get, reverse=True)
    return [docmap[key] for key in order]


def rag(query):

    # one RAGObserve trace per question; every stage logged via the SDK
    with ragobserve.trace("query", query=query):

        vec_docs = vector_retriever.invoke(query)
        bm_docs = bm25_retriever.invoke(query)
        ragobserve.log_retrieval(query, [{"text": d.page_content} for d in vec_docs],
                               retriever="faiss")
        ragobserve.log_retrieval(query, [{"text": d.page_content} for d in bm_docs],
                               retriever="bm25")

        fused = reciprocal_rank_fusion([vec_docs, bm_docs])
        ragobserve.log_fusion(
            [{"text": d.page_content} for d in fused],
            inputs={
                "vector": [{"text": d.page_content} for d in vec_docs],
                "bm25": [{"text": d.page_content} for d in bm_docs],
            },
            strategy="reciprocal rank fusion",
        )

        scores = reranker.score([(query, d.page_content) for d in fused])
        ranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:3]
        ragobserve.log_rerank(
            before=[{"text": d.page_content} for d in fused],
            after=[{"text": d.page_content, "score": float(s)} for d, s in ranked],
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        docs = [d for d, _ in ranked]
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

        ragobserve.log_context(prompt, query=query,
                             chunks=[{"text": d.page_content} for d in docs],
                             context_window=8192)

        t0 = time.time()
        response = llm.invoke(prompt)
        usage = getattr(response, "usage_metadata", None) or {}
        ragobserve.log_generation(
            model="llama-3.1-8b-instant", prompt=prompt, response=response.content,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            duration_ms=(time.time() - t0) * 1000.0, provider="groq",
        )

        return response.content, context


with gr.Blocks() as demo:

    q = gr.Textbox(label="Question")

    answer = gr.Textbox(lines=8)
    context = gr.Textbox(lines=8)

    btn = gr.Button("Ask")

    btn.click(
        rag,
        q,
        [answer, context]
    )

demo.launch()
