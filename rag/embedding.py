from sentence_transformers import SentenceTransformer
import numpy as np
from .config import EMBEDDING_MODEL

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def encode_texts(texts):
    embs = get_embedder().encode(texts, normalize_embeddings=True)
    return np.array(embs, dtype="float32")
