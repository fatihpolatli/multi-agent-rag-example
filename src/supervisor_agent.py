import sqlite3
import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.sqlite import SqliteStore

import config
from rag_agent import RagAgent

# Module-level variable to store the rag_agent instance
_rag_agent_instance = None

@dataclass
class Context:
    user_id: str

# Define tool function OUTSIDE the class - uses module-level rag_agent
@tool
def rag_search(query: str):
    """Perform a RAG search over the document store."""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        return []
    results = _rag_agent_instance.get_rag_agent().invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return results


class SupervisorAgent:
    rag_agent = None
    supervisor_agent = None
    model = None
    LLM_MODEL = config.LLM_MODEL
    MODEL_PROVIDER = config.MODEL_PROVIDER

    SUPERVISOR_PROMPT = config.SUPERVISOR_PROMPT

    def __init__(self):
        print("Supervisor Agent initialized")
        self.rag_agent = RagAgent()
        global _rag_agent_instance
        _rag_agent_instance = self.rag_agent  # Set module-level rag_agent
        

    def init_agent(self):
        if self.supervisor_agent is not None:
            return self.supervisor_agent
        conn = sqlite3.connect(config.AGENT_MEMORY_DB, check_same_thread=False)
        store = SqliteStore(conn)
        store.setup()
        set_llm_cache(SQLiteCache(database_path=config.AGENT_MEMORY_CACHE_DB))
        llm = self.get_model()
        #toolkit = SQLDatabaseToolkit(db=store, llm=llm)
        #tools = toolkit.get_tools()
        #tools.append(rag_search)

        self.supervisor_agent = create_agent(
            llm,
            tools=[rag_search],
            system_prompt=self.SUPERVISOR_PROMPT,
            store=store,
            checkpointer=InMemorySaver(),
            context_schema=Context
        )
        return self.supervisor_agent

  

    def get_model(self):
        if self.model is not None:
            return self.model
        
        self.model = init_chat_model(self.LLM_MODEL, model_provider=self.MODEL_PROVIDER, kwargs={"temperature": 0.5})
        return self.model
    

    def send_query(self, query, user_id=""):
        print(f"Sending query {query}")
        if self.supervisor_agent is None:
            self.init_agent()
        result= ""
        unique_id = uuid.uuid4()

        if self.supervisor_agent is not None:
            '''
            
            for step in self.supervisor_agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": user_id},"run_id": unique_id},
                context=Context(user_id=user_id),
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        message.pretty_print()
                        if isinstance(message, AIMessage):
                            result = message.content
            '''
            result = self.supervisor_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": user_id},"run_id": unique_id},
                context=Context(user_id=user_id),
            )
        return result