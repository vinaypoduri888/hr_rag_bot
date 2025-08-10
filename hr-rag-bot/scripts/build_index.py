import json, os
import numpy as np
import faiss
from rag.embedding import encode_texts
from rag.config import DATA_PATH, INDEX_PATH, META_PATH

def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["employees"]

def build_corpus_row(e):
    return f"{e['name']} | skills: {', '.join(e['skills'])} | exp: {e['experience_years']} years | projects: {', '.join(e['projects'])} | availability: {e['availability']}"

if __name__ == "__main__":
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    employees = load_data()
    texts = [build_corpus_row(e) for e in employees]
    X = encode_texts(texts)  # normalized float32

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine via inner product on normalized vectors
    index.add(X)

    faiss.write_index(index, INDEX_PATH)

    meta = {
        "vec_id_to_emp_id": {str(i): e["id"] for i, e in enumerate(employees)},
        "vec_id_to_text": {str(i): texts[i] for i in range(len(texts))},
        "employees": { e["id"]: e for e in employees }
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Built index -> {INDEX_PATH} and meta -> {META_PATH}")
