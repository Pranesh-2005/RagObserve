import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings
)

from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever
)

from llama_index.llms.groq import Groq
from llama_index.core.postprocessor import (
    SentenceTransformerRerank
)
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer

import ragobserve
from ragobserve.adapters.llamaindex import register, instrument_postprocessor


# local embedding model so the pipeline runs offline / Groq-only.
# (LlamaIndex otherwise defaults to OpenAIEmbedding and needs OPENAI_API_KEY.)
# Implemented directly on sentence-transformers to avoid an extra
# llama-index-embeddings-* dependency.
class LocalEmbedding(BaseEmbedding):
    _model: SentenceTransformer

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", **kw):
        super().__init__(model_name=model_name, **kw)
        object.__setattr__(self, "_model", SentenceTransformer(model_name))

    def _embed(self, text):
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def _get_query_embedding(self, query):
        return self._embed(query)

    def _get_text_embedding(self, text):
        return self._embed(text)

    async def _aget_query_embedding(self, query):
        return self._embed(query)


Settings.embed_model = LocalEmbedding()

ragobserve.init(project="llamaindex-rag")
# one call instruments the global dispatcher: node-parsing + embedding (ingest),
# retrieval, rerank and generation are all captured automatically
_rs = register()

DOCS = [
    "Retrieval Augmented Generation combines retrieval with LLM generation.",
    "BM25 is a lexical search algorithm based on term frequency and inverse document frequency.",
    "Vector search uses embeddings to find semantically similar documents.",
    "Reranking improves retrieval quality by rescoring candidate documents.",
    "Groq provides fast inference for open-source large language models.",
    "Hybrid search combines lexical and semantic retrieval methods."
]

documents = [
    Document(text=d)
    for d in DOCS
]

llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# QueryFusionRetriever resolves Settings.llm at construction, so wire Groq in
# BEFORE building it — otherwise it falls back to the OpenAI default.
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents
)

vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever],
    similarity_top_k=5,
    num_queries=1,
    llm=llm
)

# SentenceTransformerRerank emits no ReRank event, so wrap it to auto-log
reranker = instrument_postprocessor(
    SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3
    )
)

def rag(query):

    # no query engine here, so reset the handler's trace id to start a fresh
    # RAGObserve trace for each question
    _rs._state["trace_id"] = None

    nodes = fusion_retriever.retrieve(
        query
    )

    nodes = reranker.postprocess_nodes(
        nodes,
        query_str=query
    )

    context = "\n\n".join(
        n.text
        for n in nodes
    )

    prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.complete(
        prompt
    )

    return (
        str(response),
        context
    )

with gr.Blocks() as demo:

    q = gr.Textbox()

    answer = gr.Textbox(lines=8)
    context = gr.Textbox(lines=8)

    btn = gr.Button("Ask")

    btn.click(
        rag,
        q,
        [answer, context]
    )

demo.launch()