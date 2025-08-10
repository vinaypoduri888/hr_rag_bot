import google.generativeai as genai
from typing import List
from .config import GEMINI_API_KEY

def get_gemini():
    if not GEMINI_API_KEY:
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-pro")

SYSTEM_STYLE = """You are an HR assistant. 
Return concise, helpful recommendations. 
Cite concrete skills, years, project/domain signals, and availability. 
Format with short bullet points for each candidate followed by a one-paragraph recommendation."""

def build_prompt(user_query: str, items: List[dict]) -> str:
    lines = [SYSTEM_STYLE, "", f"HR Query: {user_query}", "", "Top candidates:"]
    for i, it in enumerate(items, 1):
        e = it["employee"]
        lines.append(
            f"{i}. {e['name']} — {e['experience_years']} yrs; skills={', '.join(e['skills'])}; "
            f"projects={', '.join(e['projects'])}; availability={e['availability']}; "
            f"reasons={', '.join(it.get('reasons', []))}"
        )
    lines.append("")
    lines.append("Now produce a natural response with rationale and suggestions.")
    return "\n".join(lines)

def generate_answer(user_query: str, retrieved_items):
    if not GEMINI_API_KEY:
        bullets = []
        for it in retrieved_items:
            e = it.employee
            bullets.append(
                f"- {e.name} ({e.experience_years} yrs) — skills: {', '.join(e.skills)}; "
                f"projects: {', '.join(e.projects)}; availability: {e.availability}"
            )
        text = "\n".join(bullets) or "No candidates found."
        return f"(Gemini disabled) Candidates:\n{text}"

    model = get_gemini()
    items = []
    for it in retrieved_items:
        items.append({
            "employee": it.employee.model_dump(),
            "score": it.score,
            "reasons": it.reasons
        })
    prompt = build_prompt(user_query, items)
    resp = model.generate_content(prompt)
    return resp.text.strip()
