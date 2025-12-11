import os
import re
from chromadb import Client, PersistentClient, EphemeralClient
from sentence_transformers import SentenceTransformer, CrossEncoder

class RAGEngine:
    def __init__(self):
        print("--- RAG Engine: Initializing... ---")
        
        # --- CLIENT SETUP ---
        # We use PersistentClient to store data in /tmp (writable in Cloud Run)
        # If that fails, we fall back to Ephemeral (RAM-only)
        try:
            self.client = PersistentClient(path="/tmp/chroma_db")
            print("--- ChromaDB Client Created Successfully in /tmp ---")
        except Exception as e:
            print(f"--- ChromaDB Init Failed: {e} ---")
            print("--- Switching to EphemeralClient (RAM only) ---")
            self.client = EphemeralClient()

        # Fresh collection
        self.collection = self.client.get_or_create_collection("nigeria_legal_db_v6")
        
        # 1. Retriever (Fast)
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Reranker (Smart)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.is_loaded = False
        print("--- RAG Engine: Initialization Complete. ---")

    def clean_text(self, text):
        """Cleans [source] tags and weird formatting."""
        # Remove tags if they exist
        text = re.sub(r'\\', '', text)
        text = text.replace('&nbsp;', ' ')
        # Fix multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def chunk_constitution(self, text):
        """Splits Constitution by 'Section X.'"""
        pattern = r'(Section\s+\d+\..*?)(?=\nSection\s+\d+\.|$)'
        raw_chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = []
        for chunk in raw_chunks:
            clean = chunk.strip()
            if len(clean) > 30: 
                formatted_chunks.append(f"Constitution of Nigeria 1999: {clean}")
        return formatted_chunks

    def chunk_police_act(self, text):
        """Splits Police Act by numbered sections."""
        pattern = r'(\n\d+\.\s+.*?)(?=\n\d+\.\s+|$)'
        raw_chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = []
        for chunk in raw_chunks:
            clean = chunk.strip()
            if len(clean) > 50:
                formatted_chunks.append(f"Nigeria Police Act: Section {clean}")
        return formatted_chunks

    def chunk_tenancy_law(self, text):
        """
        Splits Lagos Tenancy Law.
        Handles formats like: '1.-(1) This Law...' or '3. A tenancy...'
        """
        # Pattern looks for: newline + digits + (dot OR dot-hyphen-paren)
        pattern = r'(\n\d+(?:[\.-].*?)?.*?)(?=\n\d+(?:[\.-].*?)?|$)'
        raw_chunks = re.findall(pattern, text, re.DOTALL)
        
        formatted_chunks = []
        for chunk in raw_chunks:
            clean = chunk.strip()
            # Filter out Table of Contents or empty lines
            if len(clean) > 50 and "Arrangement of Sections" not in clean:
                formatted_chunks.append(f"Lagos Tenancy Law 2011: Section {clean}")
        return formatted_chunks

    def load_constitution(self):
        if self.is_loaded:
            return

        print("--- RAG Engine: Loading Documents... ---")

        # 1. Load Constitution
        const_path = "data/Constitution of the Federal Republic of Nigeria.txt"
        if os.path.exists(const_path):
            print(f"--- Processing {const_path}... ---")
            with open(const_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = self.clean_text(f.read())
                chunks = self.chunk_constitution(raw_text)
                if chunks:
                    ids = [f"const_{i}" for i in range(len(chunks))]
                    embeddings = self.retriever.encode(chunks)
                    self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
                    print(f"--> Indexed {len(chunks)} Constitution sections.")

        # 2. Load Police Act
        police_path = "data/P.19.txt"
        if os.path.exists(police_path):
            print(f"--- Processing {police_path}... ---")
            with open(police_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = self.clean_text(f.read())
                chunks = self.chunk_police_act(raw_text)
                if chunks:
                    ids = [f"police_{i}" for i in range(len(chunks))]
                    embeddings = self.retriever.encode(chunks)
                    self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
                    print(f"--> Indexed {len(chunks)} Police Act sections.")

        # 3. Load Tenancy Law (NEW)
        tenancy_path = "data/Lagos Tenancy Laws.txt"
        if os.path.exists(tenancy_path):
            print(f"--- Processing {tenancy_path}... ---")
            with open(tenancy_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = self.clean_text(f.read())
                chunks = self.chunk_tenancy_law(raw_text)
                if chunks:
                    ids = [f"tenancy_{i}" for i in range(len(chunks))]
                    embeddings = self.retriever.encode(chunks)
                    self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
                    print(f"--> Indexed {len(chunks)} Tenancy Law sections.")
        else:
            print(f"--- Warning: {tenancy_path} not found. Skipping. ---")

        self.is_loaded = True

    def query_law(self, question: str, initial_k=15, final_k=3):
        if not self.is_loaded:
            self.load_constitution()
        
        print(f"--- [Step 1] Retrieving top {initial_k} candidates for: '{question}' ---")
        query_embedding = self.retriever.encode([question])
        
        # 1. Broad Search
        results = self.collection.query(
            query_embeddings=query_embedding, 
            n_results=initial_k
        )
        
        if not results['documents']:
            return []

        candidates = results['documents'][0]
        
        # 2. Reranking
        print(f"--- [Step 2] Reranking candidates... ---")
        pairs = [[question, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score descending
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # 3. Select Top K
        top_results = []
        for doc, score in scored_candidates[:final_k]:
            print(f"   -> Score {score:.4f}: {doc[:50]}...")
            top_results.append(doc)
            
        return top_results