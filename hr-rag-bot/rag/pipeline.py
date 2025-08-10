from .store import VectorStore
from .retriever import hybrid_retrieve
from .generator import generate_answer
from .models import ChatResponse

_store = None

def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store

def rag_chat(message: str, top_k: int = 5) -> ChatResponse:
    store = get_store()
    retrieved, debug = hybrid_retrieve(message, store, top_k=top_k)
    answer = generate_answer(message, retrieved)
    return ChatResponse(
        answer=answer,
        results=retrieved,
        used_hybrid=True,
        debug=debug
    )
