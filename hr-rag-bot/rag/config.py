from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATA_PATH = os.getenv("DATA_PATH", "./data/employee_data.json")
INDEX_PATH = os.getenv("INDEX_PATH", "./data/index.faiss")
META_PATH  = os.getenv("META_PATH", "./data/meta.json")

TOP_K = int(os.getenv("TOP_K", "5"))
