import argparse
import os

from supervisor_agent import SupervisorAgent
class QueryCaller:
    supervisor_agent = None
    def __init__(self):
        self.supervisor_agent = SupervisorAgent()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Caller for RAG Agent")
    parser.add_argument("--query", type=str, required=True, help="The query to send to the RAG agent")
    parser.add_argument("--model", type=str, required=False, help="The model to use for the RAG agent (default: granite4:latest)", default="granite4:latest")
    parser.add_argument("--embedding_model", type=str, required=False, help="The model to use for the RAG agent (default: qwen3-embedding:0.6b)", default="qwen3-embedding:0.6b")

    args = parser.parse_args()

    os.environ["LLM_MODEL"] = args.model
    os.environ["EMBEDDING_MODEL"] = args.embedding_model

    query_caller = QueryCaller()
    response = query_caller.supervisor_agent.send_query(args.query)
    print("Response from RAG Agent:")
    print(response)