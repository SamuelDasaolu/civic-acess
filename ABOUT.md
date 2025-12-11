# Civic Access: Democratizing Access to Nigerian Law

### 🛑 The Problem
The Nigerian Constitution is the supreme law of the land, yet millions of citizens cannot access it effectively due to **complex legal jargon** and **language barriers**. When a citizen encounters a police check or a tenancy dispute, they don't need a textbook—they need an answer in the language they speak and understand.

### 💡 The Solution
**Civic Access** is a multi-lingual, RAG-powered legal assistant that bridges this gap. It doesn't just "chat"; it performs **Retrieve-and-Rerank** searches across the **1999 Constitution**, **Police Act**, and **Lagos Tenancy Laws**, ensuring every answer is grounded in real legal text.

### ✨ Key Innovations
* **🗣️ Hyper-Localized AI:** It speaks **Nigerian Pidgin**, **Yoruba**, **Hausa**, and **Igbo**. It adopts a persona (e.g., a "Lagos Street" persona for Pidgin users) to make the law relatable.
* **⚖️ The "Lazy Judge" System:** We built an automated evaluation pipeline. Every interaction is asynchronously graded by a second AI (Gemini 1.5 Flash) to detect hallucinations and ensure legal accuracy without slowing down the user experience.
* **🔄 Semantic Translation Layer:** To solve the "low-resource language" problem in vector search, we implemented a real-time translation layer that converts dialect-heavy queries (e.g., *"Wetin be my right?"*) into formal legal English query vectors, ensuring high-precision retrieval.