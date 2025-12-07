import os
import re
from chromadb import Client
from sentence_transformers import SentenceTransformer, CrossEncoder # <--- NEW IMPORT

class RAGEngine:
    def __init__(self):
        print("--- RAG Engine: Initializing... ---")
        self.client = Client()
        # Fresh collection to ensure clean data
        self.collection = self.client.get_or_create_collection("nigeria_legal_db_smart")
        
        # 1. The Retriever (Fast, finds candidates)
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. The Reranker (Smart, sorts candidates)
        # This model is specifically trained to grade Q&A relevance
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.is_loaded = False
        print("--- RAG Engine: Initialization Complete. ---")

    def clean_text(self, text):
        text = re.sub(r'\\', '', text)
        text = text.replace('&nbsp;', ' ')
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def chunk_constitution(self, text):
        pattern = r'(Section\s+\d+\..*?)(?=\nSection\s+\d+\.|$)'
        raw_chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = []
        for chunk in raw_chunks:
            clean = chunk.strip()
            if len(clean) > 30: 
                # Stronger Context Injection
                formatted_chunks.append(f"Constitution of Nigeria 1999: {clean}")
        return formatted_chunks

    def chunk_police_act(self, text):
        pattern = r'(\n\d+\.\s+.*?)(?=\n\d+\.\s+|$)'
        raw_chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = []
        for chunk in raw_chunks:
            clean = chunk.strip()
            if len(clean) > 50:
                formatted_chunks.append(f"Nigeria Police Act: Section {clean}")
        return formatted_chunks

    def load_constitution(self):
        if self.is_loaded:
            return

        # Load Constitution
        const_path = "Constitution of the Federal Republic of Nigeria.txt"
        if os.path.exists(const_path):
            print(f"--- Processing {const_path}... ---")
            with open(const_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = self.clean_text(f.read())
                const_chunks = self.chunk_constitution(raw_text)
                if const_chunks:
                    ids = [f"const_{i}" for i in range(len(const_chunks))]
                    # Embed using the Retriever
                    embeddings = self.retriever.encode(const_chunks)
                    self.collection.add(embeddings=embeddings, documents=const_chunks, ids=ids)
                    print(f"--> Indexed {len(const_chunks)} Constitution sections.")

        # Load Police Act
        police_path = "P.19.txt"
        if os.path.exists(police_path):
            print(f"--- Processing {police_path}... ---")
            with open(police_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = self.clean_text(f.read())
                police_chunks = self.chunk_police_act(raw_text)
                if police_chunks:
                    ids = [f"police_{i}" for i in range(len(police_chunks))]
                    embeddings = self.retriever.encode(police_chunks)
                    self.collection.add(embeddings=embeddings, documents=police_chunks, ids=ids)
                    print(f"--> Indexed {len(police_chunks)} Police Act sections.")

        self.is_loaded = True

    def query_law(self, question: str, initial_k=15, final_k=3):
        """
        Two-Step Search:
        1. Retrieve top 15 candidates using Vectors (Fast)
        2. Rerank them using Cross-Encoder (Smart)
        3. Return top 3
        """
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
        
        # 2. Reranking (The Logic Fix)
        print(f"--- [Step 2] Reranking candidates... ---")
        
        # Prepare pairs for the Cross-Encoder: [[Question, Candidate1], [Question, Candidate2], ...]
        pairs = [[question, doc] for doc in candidates]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Zip texts with scores and sort by score descending
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # 3. Select Top K
        top_results = []
        for doc, score in scored_candidates[:final_k]:
            print(f"   -> Score {score:.4f}: {doc[:50]}...")
            top_results.append(doc)
            
        return top_results


