import os

from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "granite4:latest")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "4096"))

# Vector Store
MILVUS_URI_PATH = os.getenv("MILVUS_URI_PATH", "./milvus_demo.db")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test")

# Memory
AGENT_MEMORY_DB = os.getenv("AGENT_MEMORY_DB", "agent_memory.db")
AGENT_MEMORY_CACHE_DB = os.getenv("AGENT_MEMORY_CACHE_DB", "agent_memory_cache.db")

# API
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")

# Prompts
SUPERVISOR_PROMPT = os.getenv("SUPERVISOR_PROMPT", "You are a helpful personal assistant. If user ask question, you can make search on RAG knowledge base to answer user queries if neccessary. If you cannot find the answer in RAG knowledge base, you will say 'you have not knowledge about this topic' instead of making up an answer.")
RAG_PROMPT = os.getenv("RAG_PROMPT", "You are a research assistant. Use similarity_search to find relevant information from the document store to answer user queries. If you cannot find the answer in RAG knowledge base, answer that you don't know the answer instead of making up an answer.")
