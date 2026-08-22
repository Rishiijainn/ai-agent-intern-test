import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
#project paths
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
DATA_DIR = BASE_DIR / "data"
EVALUATION_DIR = BASE_DIR / "evaluation"
ORDERS_FILE = DATA_DIR / "orders.json"

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

#agent Defaults

TOP_K_RETRIEVAL = 3
MAX_CONVERSATION_TURNS = 6