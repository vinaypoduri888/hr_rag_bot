import re
from typing import List, Dict, Tuple
import numpy as np
from .embedding import encode_texts
from .store import VectorStore
from .models import RetrievedItem, Employee
from .config import TOP_K

SKILL_WORDS = set([
    "python","java","aws","docker","react","react native","pytorch","tensorflow",
    "ml","machine learning","nlp","kubernetes","gcp","azure","sql","node","go",
    "scala","spark","pandas","scikit-learn","flask","fastapi"
])

def parse_query(query: str) -> Dict:
    q = query.lower()

    years = None
    m = re.search(r'(\d+)\s*\+?\s*(?:years|yrs|y)', q)
    if m:
        years = int(m.group(1))

    skills = []
    for w in SKILL_WORDS:
        if w in q:
            skills.append(w)

    availability = None
    if "available" in q:
        availability = "available"

    domain = None
    for d in ["healthcare","fintech","e-commerce","ecommerce","gaming","education","devops"]:
        if d in q:
            domain = d
            break

    return {"years": years, "skills": skills, "availability": availability, "domain": domain}

def hybrid_retrieve(query: str, store: VectorStore, top_k: int = TOP_K) -> Tuple[List[RetrievedItem], Dict]:
    dense_vec = encode_texts([query])
    dists, idxs = store.search(dense_vec, max(top_k*3, top_k))
    candidates = []
    parsed = parse_query(query)

    for dist, idx in zip(dists, idxs):
        if idx == -1:
            continue
        emp_raw = store.get_employee_by_idx(int(idx))
        emp = Employee(**emp_raw)
        text = store.get_text_by_idx(int(idx))

        reasons = []
        score = float(1.0 - dist)

        for s in parsed["skills"] or []:
            if any(s in k.lower() for k in emp.skills):
                score += 0.05
                reasons.append(f"skill:{s}")

        if parsed["availability"] == "available" and emp.availability.lower() == "available":
            score += 0.05
            reasons.append("availability:available")

        if parsed["years"] is not None and emp.experience_years >= parsed["years"]:
            score += 0.07
            reasons.append(f"years>={parsed['years']}")

        if parsed["domain"]:
            joined_projects = " ".join([p.lower() for p in emp.projects])
            if parsed["domain"] in joined_projects:
                score += 0.06
                reasons.append(f"domain:{parsed['domain']}")

        candidates.append((score, emp, reasons))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = [RetrievedItem(employee=e, score=round(s, 4), reasons=r) for s, e, r in candidates[:top_k]]
    debug = {"parsed_query": parsed, "raw_hits": len(candidates)}
    return out, debug
