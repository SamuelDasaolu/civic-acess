# 🏛️ Civic-Access (Project by Team SABILAW)

> **Status:** Hackathon Submission  
> **Live API:** `https://civic-backend-528126792252.us-east4.run.app/docs`  
> **Frontend:** `https://civic-access.web.app`

## 📖 Overview
**Civic Access** is a high-performance FastAPI backend designed to democratize legal access in Nigeria. It leverages **Retrieval-Augmented Generation (RAG)** to provide accurate legal citations from the **1999 Constitution**, **Police Act**, and **Lagos Tenancy Laws**.

Crucially, it breaks the language barrier by supporting **Nigerian Pidgin, Yoruba, Hausa, and Igbo**, ensuring that the law is accessible to everyone, not just the elite.

## 🏗️ Architecture

The system follows a **Microservices** pattern hosted on **Google Cloud Platform**:

1.  **Frontend (React/Firebase):** Captures user intent and preferred dialect.
2.  **Translation Layer (Google Cloud Translate):** Auto-detects dialects (e.g., *"Wetin be the law?"*) and normalizes them to English for effective Vector Search.
3.  **RAG Engine (ChromaDB + SentenceTransformers):**
    * **Retrieval:** `all-MiniLM-L6-v2` fetches top 15 legal chunks.
    * **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores them for strict relevance.
4.  **Inference (NCAIR1/N-ATLAS model):** An 8B-parameter model hosted on **NVIDIA L4 GPUs** generates the response in the user's requested persona.
5.  **Persistence (Turso):** User interactions and logs are stored in a distributed SQLite database (**Turso**) to survive Cloud Run's ephemeral restarts.
6.  **Evaluation (Gemini 2.0 Flash):** An asynchronous background task acts as a "Judge," grading every answer for accuracy and hallucination.

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python 3.11)
* **Compute:** Google Cloud Run (Serverless, Min Instances: 1, CPU Boost enabled)
* **AI/LLM:** NCAIR1/N-ATLAS (Custom Container) + Gemini 1.5 Flash (Judge)
* **Vector DB:** ChromaDB (Persistent Client in `/tmp`)
* **Database:** Turso (LibSQL)
* **Translation:** Google Cloud Translation API (v2)
* **CI/CD:** GitHub Actions -> Google Artifact Registry -> Cloud Run

## 🚀 Key Features & Architectural Challenges Solved

Building **N-Atlas** wasn't just about connecting APIs. We faced significant hurdles in deploying a specialized Legal AI for a multi-lingual audience. Here is how we solved them:

### 1. The "Tower of Babel" Vector Mismatch
* **The Problem:** Our vector database contains the Nigerian Constitution in **English**. When a user asked a question in **Pidgin** (*"Wetin be the law for tiff?"*) or **Yoruba**, the semantic distance between the query and the documents was too large. The RAG engine returned irrelevant chunks, causing the model to hallucinate.
* **The Strategy:** We implemented a **"Hybrid Translation Pipeline."**
    * **Step 1 (Ingest):** User query is auto-detected and translated to English using the **Google Cloud Translation API** (v2).
    * **Step 2 (Retrieve):** The *translated* English query searches the vector database to find the accurate legal section.
    * **Step 3 (Generate):** The *original* Pidgin/Yoruba query + the *English* legal context are fed to the LLM. This forces the model to "think" in Law but "speak" in Street Persona.

### 2. The "Stateless" Amnesia (Cloud Run Persistence)
* **The Problem:** We chose **Google Cloud Run** for its serverless scalability, but its filesystem is ephemeral. Every time the server idled or redeployed, our local `users.db` and `chat_logs.db` were wiped instantly. This made maintaining user accounts impossible.
* **The Strategy:** We moved to a **Distributed Database Architecture**.
    * **Logs & Auth:** We migrated from local SQLite to **Turso (LibSQL)**, a remote edge database that persists data independently of our container's lifecycle.
    * **Vector Index:** We re-architected the RAG engine to initialize the ChromaDB client in `/tmp` (the only writable directory in Cloud Run) and implemented a fallback to **Ephemeral (RAM) mode** to prevent crashes during cold boots.

### 3. The "Needle in a Haystack" (Retrieval Precision)
* **The Problem:** Standard "fixed-size chunking" (splitting text every 500 characters) failed miserably for legal texts. It would cut a law in half, separating the "Crime" from the "Penalty," confusing the AI.
* **The Strategy:** We wrote **Regex-Based Smart Chunking**.
    * Instead of character counts, our engine splits documents by **Legal Sections** (e.g., regex pattern `r'(Section\s+\d+\..*?)'`).
    * **Metadata Injection:** We prepended the document source to every chunk (e.g., *"Constitution of Nigeria 1999: Section 33..."*) so the embeddings explicitly capture the source authority.
    * **Reranking:** We added a **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** step. After retrieving the top 15 broad results, the Reranker strictly grades them and discards the noise, keeping only the top 3 highly relevant laws.

### 4. The "Cold Start" Latency
* **The Problem:** Booting a container with **PyTorch, SentenceTransformers, and ChromaDB** took 20-30 seconds. This made the app feel "broken" to the first user.
* **The Strategy:**
    * We enabled **CPU Boost** on Cloud Run to overdrive the processor during startup.
    * We configured **`min-instances: 1`** for demo periods, keeping the heavy RAG engine loaded in memory to ensure sub-second response times.

### 5. Trust & Verification (The "Lazy Judge")
* **The Problem:** How do we know if the 8B model is giving accurate legal advice or just making things up?
* **The Strategy:** We built an **Asynchronous Evaluation Pipeline**.
    * User requests are handled immediately.
    * In the background (`BackgroundTasks`), a separate call is sent to **Gemini 1.5

## 🔧 Installation & Local Setup

**Prerequisites:** Python 3.11+, Google Cloud SDK, Turso Account.

```bash
# 1. Clone the repo
git clone [https://github.com/your-username/civic-backend.git](https://github.com/SamuelDasaolu/civic-acess.git)
cd civic-backend/backend

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Environment Variables (.env)
# To get VERTEX_RELATED tokens, first deploy the NCAIR1/N-ATLAS model found on HugginFace on Google's Vertex Servers
# Create a .env file with:
VERTEX_PROJECT_ID=your-project-id
VERTEX_LOCATION=us-east4
VERTEX_ENDPOINT_ID=your-endpoint-id
GEMINI_API_KEY=your-gemini-key
TURSO_URL=libsql://your-db.turso.io
TURSO_TOKEN=your-turso-token

# 4. Run Locally
uvicorn main:app --reload