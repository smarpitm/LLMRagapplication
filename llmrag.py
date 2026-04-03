import os
import uuid
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dataprep import *

#rag reterival pipeline from vector store 



class RAGRetrieval:
    def __init__(self, vector_store: VectorStoreManager, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        """
        print(f"Retrieving documents for query: '{query}'")
        
        # 1. Generate query embedding
        # We wrap query in a list [query] because the model expects a batch
        # We take [0] because we only have one query
        query_embedding_np = self.embedding_manager.generate_embeddings([query])[0]
        
        # Convert numpy array to python list (Required by ChromaDB)
        query_embedding_list = query_embedding_np.tolist()

        # 2. Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                # We explicitly request documents, metadata, and distances
                include=["documents", "metadatas", "distances"]
            )
            
            # 3. Format results
            retrieved_docs = []
            
            # Chroma returns a list of lists (batch format). 
            # Since we sent 1 query, we look at index 0 for all fields.
            if results['ids'] and results['ids'][0]:
                num_results = len(results['ids'][0])
                
                for i in range(num_results):
                    distance = results['distances'][0][i]
                    
                   
                    doc_data = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": distance, # Distance metric (e.g., L2 or Cosine distance)
                        "id": results['ids'][0][i]
                    }
                    retrieved_docs.append(doc_data)
            
            print(f"Found {len(retrieved_docs)} relevant documents.")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# --- Usage Example ---

# Initialize the Retrieval Pipeline
rag = RAGRetrieval(vector_store, embedding_manager)

# Test a query
if __name__ == "__main__":      
    
    query = "What is the conclusion of the document?"
    results = rag.retrieve(query, top_k=3)

    # Print results
    print("\n--- Search Results ---")
    for doc in results:
        print(f"\n[Score: {doc['score']:.4f}]")
        print(f"Content: {doc['content'][:200]}...") # Print first 200 chars


#next steps: integrate with LLM for answer generation   
 #integration vector db context pipeline with llm for answer generation
 #simple example using langchain-groq

GROQ_API_KEY = "apikey" #
# Initialize ChatGroq

llm = ChatGroq(
    api_key=GROQ_API_KEY, 
    model="llama-3.1-8b-instant", 
    temperature=0
)

#rag pipeline function
def rag_simple(query: str, retriever: RAGRetrieval, llm, top_k: int = 3):
    
    # 1. Retrieve Docs
    print(query)
    results = retriever.retrieve(query, top_k=top_k)
    
    # 2. Build Context (If any)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    

    # 3. Define the "Hybrid" Prompt
    
    system_instruction = (
        """ 
        First, check the 'Context' below to see if it contains the answer. 
        If the context provides the answer, use it. 
        If the context is missing, empty, or irrelevant to the question, 
        ignore it and answer using your own general knowledge."""
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    # 4. Run Chain
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"context": context, "query": query})
    
    return response


# 3. Run the Pipeline 

# Example usage
query = input("What is your question :") 
answer = rag_simple(query, rag, llm)

print("\nFinal Answer ")
print(answer)
