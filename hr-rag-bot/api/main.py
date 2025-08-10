from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from rag.pipeline import rag_chat, get_store
from rag.models import ChatQuery, ChatResponse, Employee
from rag.store import VectorStore
from rag.config import META_PATH
import json

app = FastAPI(title="HR Resource Query Chatbot (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_store = None

@app.on_event("startup")
def startup_event():
    global _store
    _store = get_store()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(q: ChatQuery):
    return rag_chat(q.message, top_k=q.top_k)

@app.get("/employees/search", response_model=List[Employee])
def employees_search(
    skill: Optional[str] = Query(default=None),
    min_years: Optional[int] = Query(default=None),
    availability: Optional[str] = Query(default=None)
):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    emps = [Employee(**e) for e in meta["employees"].values()]

    def ok(e: Employee) -> bool:
        if skill and not any(skill.lower() in s.lower() for s in e.skills):
            return False
        if min_years is not None and e.experience_years < min_years:
            return False
        if availability and e.availability.lower() != availability.lower():
            return False
        return True

    return [e for e in emps if ok(e)]
