import json
import faiss
import numpy as np
from typing import List, Tuple
from .config import INDEX_PATH, META_PATH

class VectorStore:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta = None

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def is_ready(self) -> bool:
        return self.index is not None and self.meta is not None

    def search(self, query_vec: np.ndarray, top_k: int):
        distances, indices = self.index.search(query_vec, top_k)
        return distances[0], indices[0]

    def get_employee_by_idx(self, idx: int) -> dict:
        emp_id = self.meta["vec_id_to_emp_id"][str(idx)]
        return self.meta["employees"][emp_id]

    def get_text_by_idx(self, idx: int) -> str:
        return self.meta["vec_id_to_text"][str(idx)]
