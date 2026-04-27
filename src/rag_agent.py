from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document

import config
from vector_store_manager import VectorStoreManager


# Module-level variable to store the vector_store_manager instance
_vector_store_manager = None


# Define tool function OUTSIDE the class - uses module-level vector_store_manager
@tool("search_similar_documents")
def search_similar_documents(query: str, top_k: int = 5) -> list[Document]:
    """Search for similar documents in the vector store."""
    global _vector_store_manager
    if _vector_store_manager is None:
        return []
    results = _vector_store_manager.search_similar_documents(query, top_k)
    return results


class RagAgent:
    vector_store_manager = None
    rag_agent = None
    model = None
    LLM_MODEL = config.LLM_MODEL
    MODEL_PROVIDER = config.MODEL_PROVIDER

    SYSTEM_PROMPT = config.RAG_PROMPT
    

    def __init__(self):
        print("RAG Agent initialized")
        self.vector_store_manager = VectorStoreManager()
        global _vector_store_manager
        _vector_store_manager = self.vector_store_manager

    def get_rag_agent(self):
        if self.rag_agent is not None:
            return self.rag_agent

        self.rag_agent = create_agent(
            self.get_model(), 
            tools=[search_similar_documents], 
            system_prompt=self.SYSTEM_PROMPT
        )
        return self.rag_agent
    
    def get_model(self):
        if self.model is not None:
            return self.model
        
        self.model = init_chat_model(self.LLM_MODEL, model_provider=self.MODEL_PROVIDER)
        return self.model