import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

import config

class VectorStoreManager:

    URI = config.MILVUS_URI_PATH
    MODEL = config.EMBEDDING_MODEL
    COLLECTION_NAME = config.MILVUS_COLLECTION_NAME

    def __init__(self):
        
        print("Custom Agent initialized")

        self.init_vector_store()

    def init_vector_store(self):

        
        self.vector_store = Milvus(collection_name=self.COLLECTION_NAME,auto_id=True, embedding_function=self.get_embeddings(),  connection_args={"uri": self.URI},
                        index_params={"index_type": "IVF_FLAT", "metric_type": "L2"})
    
    def add_documents_to_vector_store(self, file_path):
        if self.vector_store is None or len(file_path) == 0:
            return
        

        documents = self.load_and_process_pdf_documents(file_path)
        self.vector_store.add_documents(documents=documents)
        
    def get_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.MODEL, base_url=config.OLLAMA_BASE_URL, dimensions=config.EMBEDDING_DIMENSIONS)

    def clean_text(self, text) -> str:
        """Cleans and normalizes text."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except common punctuation
        return text.strip()
    
    def load_and_process_pdf_documents(self, file_path) -> list[Document]:
        # Implement your PDF loading and processing logic here
        # This function should return a list of processed documents
        pdf_loader = PyPDFLoader(file_path)

        pdf_docs = pdf_loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = file_path
            doc.page_content = self.clean_text(doc.page_content)

        return pdf_docs
    def load_and_process_pdf_documents_semantic(self, file_path) -> list[Document]:
        # Implement your PDF loading and processing logic here
        # This function should return a list of processed documents
        pdf_loader = PyPDFLoader(file_path)

        pdf_docs = pdf_loader.load()

        semantic_splitter = SemanticChunker(
            self.get_embeddings(), 
            breakpoint_threshold_type="percentile" 
        )

        # 3. Create chunks from your text

        for doc in pdf_docs:
            doc.metadata["source"] = file_path
            
        new_docs  = semantic_splitter.split_documents(pdf_docs)

        return new_docs
    
    def search_similar_documents(self, query, top_k=5) -> list[Document]:
        if self.vector_store is None:
            return []
        
        return self.vector_store.similarity_search(query, k=top_k, paramm={"params": {"nprobe": 10}})   