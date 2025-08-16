import os
import time
import warnings
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes

from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore


# =============================
# Carga de entorno y constantes
# =============================
warnings.filterwarnings("ignore")
load_dotenv()

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
if missing:
    print(f"⚠️ Faltan variables de entorno: {missing}")
else:
    print("✅ Credenciales OK")

# Nombre de colección en Qdrant (puede sobreescribirse por env)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "carretera_austral")


# =============================
# Clientes y modelos
# =============================
# Embeddings: la dimensión debe coincidir con la colección existente
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)

# Modelos LLM (ajusta con OPENAI_CHAT_MODEL si lo deseas)
_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=_chat_model, streaming=False, temperature=0)
llm_small = ChatOpenAI(model=_chat_model, streaming=False, temperature=0)

# Qdrant: solo conectar a colección existente (no reindexar)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
try:
    # Verificación rápida de existencia de la colección
    _ = qdrant.get_collection(COLLECTION_NAME)
    print(f"🔗 Conectando a colección existente '{COLLECTION_NAME}'...")
except Exception as e:
    raise RuntimeError(
        f"La colección '{COLLECTION_NAME}' no existe o no es accesible en Qdrant"
    ) from e

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Retriever vectorial (BM25 se aplicará localmente sobre los candidatos)
vec_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5},
)
print("✅ Vector retriever listo")


# =============================
# Tipado del estado del grafo
# =============================
class TourismRAGState(TypedDict):
    query: str
    original_query: str
    documents: List[Document]
    sources: List[str]
    doc_quality_score: float
    retrieval_attempts: int
    should_rewrite: bool
    answer: str
    workflow_steps: List[str]
    total_tokens: int


# =============================
# Utilidades
# =============================
from numpy import dot
from numpy.linalg import norm
import unicodedata
import string
import re as _re


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: List[Document]) -> List[str]:
    out: List[str] = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source_title") or f"Página {d.metadata.get('page','?')}"
        out.append(title)
    return out


def calculate_doc_relevance(docs: List[Document], query: str) -> float:
    if not docs:
        return 0.0
    try:
        q = embeddings.embed_query(query)
        doc_vecs = embeddings.embed_documents([d.page_content for d in docs])
        sims = [max(0.0, float(dot(q, v) / (norm(q) * norm(v)))) for v in doc_vecs]
        return sum(sims) / len(sims)
    except Exception:
        return 0.0


def normalize_hyphens(text: str) -> str:
    return _re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def simple_preprocess(text: str):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.replace('\n', ' ')
    table = str.maketrans({ch: ' ' for ch in '“”"' + string.punctuation})
    text = text.translate(table)
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# =============================
# Nodos del grafo
# =============================
from typing import Any as _Any


def retrieve_node(state: TourismRAGState) -> Dict[str, _Any]:
    query = state["query"]
    print(f"  🔍 Recuperando documentos para: {query}")

    # 1) Candidatos vectoriales desde Qdrant
    vec_docs = vec_retriever.invoke(query)

    # 2) Normalización para el scorer léxico
    for d in vec_docs:
        d.page_content = normalize_hyphens(d.page_content)

    # 3) BM25 on-demand SOLO sobre estos candidatos (sin corpus global)
    bm25_local = BM25Retriever.from_documents(vec_docs, k=6, preprocess_func=simple_preprocess)
    bm25_docs = bm25_local.invoke(query)

    # 4) Fusión simple preservando orden y sin duplicados
    def key(d: Document):
        return (
            d.metadata.get("_id"),
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chunk_id"),
        )

    seen = set()
    merged: List[Document] = []
    for d in bm25_docs + vec_docs:
        k = key(d)
        if k in seen:
            continue
        seen.add(k)
        merged.append(d)

    final_k = 8
    docs = merged[:final_k]

    score = calculate_doc_relevance(docs, query)
    sources = extract_sources(docs)
    steps = state.get("workflow_steps", [])
    steps.append(
        f"retrieve (2-stage): vec={len(vec_docs)} bm25={len(bm25_docs)} final={len(docs)} quality={score:.3f}"
    )
    return {
        "documents": docs,
        "doc_quality_score": score,
        "sources": sources,
        "workflow_steps": steps,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def grade_node(state: TourismRAGState) -> Dict[str, _Any]:
    score = state["doc_quality_score"]
    should_rewrite = score < 0.5
    steps = state["workflow_steps"]
    steps.append(f"grade: score={score:.3f}, rewrite={should_rewrite}")
    return {"should_rewrite": should_rewrite, "workflow_steps": steps}


def rewrite_node(state: TourismRAGState) -> Dict[str, _Any]:
    prompt = f"""
    Actúa como analista de consultas para un sistema RAG turístico sobre la Carretera Austral (Ruta 7, Patagonia, Chile).

    Objetivo: reescribir la consulta para maximizar el recall en la búsqueda semántica, preservando la intención original y el idioma.

    Instrucciones:
    - No inventes datos nuevos ni cambies la intención.
    - Explicita, cuando existan, estos elementos: tramo u origen–destino, sentido (N→S o S→N), cantidad de días/fechas/temporada, medio de transporte (auto/4x4, moto, bici, a pie, bus, ferry/barcaza), tipo de actividades (miradores, trekking, navegación, parques), restricciones (clima, niños, presupuesto, tiempo), y logística (ferries/barcazas, horarios, accesos, combustible).
    - Añade sinónimos del dominio SOLO si aumentan el recall: "Carretera Austral"/"Ruta 7"; "ferry"/"barcaza"/"transbordador"; "sendero"/"trekking"; "acceso"/"entrada"; "horario"/"itinerario"; "Parque Nacional"/"CONAF".
    - Evita vaguedad ("mejor", "lindo"); usa criterios comparables si están implícitos (distancia, tiempo, estado del camino, necesidad de 4x4).
    - Formato de salida: devuelve SOLO la consulta reescrita en UNA ÚNICA línea, sin comillas ni explicación adicional.

    Consulta original: {state['original_query']}
    Consulta actual: {state['query']}

    Consulta reescrita:
    """
    try:
        resp = llm_small.invoke(prompt)
        new_q = resp.content.strip()
        steps = state["workflow_steps"]
        steps.append(f"rewrite: '{state['query']}' -> '{new_q}'")
        return {"query": new_q, "workflow_steps": steps}
    except Exception as e:
        print(f"❌ Rewrite error: {e}")
        return {"workflow_steps": state["workflow_steps"]}


def generate_node(state: TourismRAGState) -> Dict[str, _Any]:
    docs = state["documents"]
    context = format_docs(docs)
    originally = state["original_query"]
    final_prompt = f"""
    Eres un asistente turístico especializado en la Carretera Austral (Patagonia, Chile) operando en un sistema RAG.

    Políticas de respuesta (cumple TODAS):
    1) Usa EXCLUSIVAMENTE la información del contexto proporcionado.
    2) Si la información solicitada no aparece en el contexto, responde explícitamente: "Esta información no está disponible en los documentos analizados" y solo si la consulta está relacionada con la carretera austral, sugiere qué datos faltan buscar (p. ej., horarios, temporada, accesos).
    3) Cita las fuentes cuando sea posible usando metadata disponible (página, título, fuente, sección).
    4) Prioriza utilidad práctica: tramos, tiempos/ distancias aproximadas, atractivos, accesos, estado de la ruta, clima/temporada, ferries (empalmes, requisitos, reservas), permisos (CONAF), combustible.
    5) No inventes ni extrapoles datos; no supongas horarios ni precios si no están en el contexto.
    6) Estructura la salida de forma clara: Resumen breve de la respuesta.
    7) Regla especial de consistencia geográfica y temporal: Si el contexto presenta datos de diferentes lugares o fechas, indica explícitamente a qué ubicación o periodo pertenece cada dato. No combines información de distintos periodos o lugares como si fueran del mismo.

    Ten en cuenta las siguientes notas:
    - PN es sinónimo de Parque Nacional.
    - P.N. es sinónimo de Parque Nacional.
    - MN es sinónimo de Monumento Natural.
    - M.N. es sinónimo de Monumento Natural.

    Contexto:
    {context}

    Pregunta:
    {originally}

    Respuesta:
    """
    try:
        with get_openai_callback() as cb:
            resp = llm.invoke(final_prompt)
            answer = resp.content
            steps = state["workflow_steps"]
            steps.append(f"generate: {len(answer)} chars, {cb.total_tokens} tokens")
            return {"answer": answer, "workflow_steps": steps, "total_tokens": cb.total_tokens}
    except Exception as e:
        return {"answer": f"Error generando respuesta: {e}", "workflow_steps": state["workflow_steps"]}


def decide_next(state: TourismRAGState) -> str:
    print("  🤔 Estamos mejorando la consulta...")
    if state["should_rewrite"] and state["retrieval_attempts"] <= 2:
        return "rewrite"
    return "generate"


# =============================
# Construcción del grafo
# =============================
workflow = StateGraph(TourismRAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_next, {"rewrite": "rewrite", "generate": "generate"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()
print("✅ Grafo Agentic compilado")


# =============================
# Función helper (igual a notebook, sin reindexar)
# =============================
def run_agentic_tourism_rag(query: str, verbose: bool = False) -> Dict[str, Any]:
    initial: TourismRAGState = {
        "query": query,
        "original_query": query,
        "documents": [],
        "sources": [],
        "doc_quality_score": 0.0,
        "retrieval_attempts": 0,
        "should_rewrite": False,
        "answer": "",
        "workflow_steps": [],
        "total_tokens": 0,
    }
    if verbose:
        print("\n🚀 RAG AGENTIC para turismo en la Carretera Austral")
        print(f"❓ {query}")
        print("=" * 60)
    t0 = time.time()
    result = rag_graph.invoke(initial)
    result["execution_time"] = time.time() - t0
    if verbose:
        print("\n🤖 RESPUESTA:\n" + "-" * 40)
        print(result.get("answer", ""))
        print("\n🔍 WORKFLOW:")
        for i, s in enumerate(result.get("workflow_steps", []), 1):
            print(f"  {i}. {s}")
    return result


# =============================
# Servidor LangServe (FastAPI)
# =============================
app = FastAPI(
    title="Agentic Tourism RAG - Carretera Austral",
    version="1.0",
    description="RAG agentic que consume una colección existente en Qdrant (sin reindexar)",
)


def _endpoint_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    # Admite {"query": "..."} o string directo
    if isinstance(inputs, str):
        query = inputs
    else:
        query = inputs.get("query", "")
    res = run_agentic_tourism_rag(query, verbose=False)
    return {
        "answer": res.get("answer", ""),
        "sources": res.get("sources", []),
        "workflow_steps": res.get("workflow_steps", []),
        "total_tokens": res.get("total_tokens", 0),
        "execution_time": res.get("execution_time", 0.0),
    }

_runnable = RunnableLambda(_endpoint_fn).with_types(input_type=str, output_type=Dict[str, Any])

add_routes(app, _runnable, path="/agentic_tourism")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


