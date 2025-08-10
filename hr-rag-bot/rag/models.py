from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Employee(BaseModel):
    id: str
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class ChatQuery(BaseModel):
    message: str
    top_k: int = 5

class RetrievedItem(BaseModel):
    employee: Employee
    score: float
    reasons: List[str] = []

class ChatResponse(BaseModel):
    answer: str
    results: List[RetrievedItem]
    used_hybrid: bool
    debug: Optional[Dict[str, Any]] = None
