
import os
import re
from chromadb import Client
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("--- RAG Engine: Initializing... ---")
        self.client = Client()
        # --- FINAL FIX: Use a new collection name to force a fresh index. ---
        # This guarantees that no old, corrupted data from previous attempts is used.
        self.collection = self.client.get_or_create_collection("nigerian_constitution_final")
        self.model = SentenceTransformer(model_name)
        self.is_loaded = False
        print("--- RAG Engine: Initialization Complete. ---")

    def load_constitution(self, file_path="constitution.txt"):
        if self.is_loaded:
            print("--- RAG Engine: Constitution is already loaded. ---")
            return

        print(f"--- RAG Engine: Loading constitution from {file_path}... ---")
        if not os.path.exists(file_path):
            print(f"--- RAG Engine FATAL ERROR: Constitution file not found at {file_path}. ---")
            raise FileNotFoundError(f"Constitution file not found at {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # --- Simplified Chunking Strategy: Split by Section ---
        raw_chunks = re.split(r'(Section \d+\.)', text)
        
        chunks = []
        for i in range(1, len(raw_chunks), 2):
            chunk_text = (raw_chunks[i] + raw_chunks[i+1]).strip()
            chunks.append(chunk_text)

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        print(f"--- RAG Engine: Constitution split into {len(chunks)} distinct section chunks. ---")
        
        # No need to delete; we are using a new collection name.
        embeddings = self.model.encode(chunks)
        self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        self.is_loaded = True
        print("--- RAG Engine: Constitution loaded and re-indexed successfully into new collection. ---")

    def query_law(self, question: str, n_results=1):
        if not self.is_loaded:
            self.load_constitution()
        
        print(f"--- RAG Engine: Querying for: '{question}' ---")
        query_embedding = self.model.encode([question])
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        
        retrieved_docs = results['documents'][0]
        context = "\n\n".join(retrieved_docs)
        print(f"--- RAG Engine: Retrieved {len(retrieved_docs)} documents. ---")
        return context

