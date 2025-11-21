# **StoicMind — An AI Stoic Sage Chatbot**

StoicMind is an AI-powered Stoic philosophy guide built using a Retrieval-Augmented Generation (RAG) pipeline.  
It provides personalized, Stoicism-informed reflections and guidance based on the works of Marcus Aurelius, Seneca, Epictetus, and other major Stoic thinkers.

You can access the live application here:  
👉 **http://3.22.97.87:8080**

---

## 🚀 **Project Overview**

StoicMind is designed to deliver context-aware, philosophically grounded responses using a combination of:

- **Primary Stoic texts** (Marcus Aurelius, Seneca, Epictetus, etc.)
- **A vector database** storing embeddings of all Stoic works
- **A modern LLM** (Gemini 2.5 Pro) for generation
- **A structured system prompt** that ensures responses remain faithful to Stoic principles  
  (virtue, resilience, discipline, control, framing, acceptance, etc.)

The chatbot is instructed to help users reflect on challenges, decisions, and emotional struggles through the lens of Stoic philosophy.

---

## 🧠 **Architecture & Technology Stack**

StoicMind uses a full RAG pipeline designed for efficient retrieval, high-quality embeddings, and deterministic reasoning grounded in classical Stoic texts.

### **Core Technologies**

- **LangChain** — orchestration framework for RAG, retrieval, and prompt routing  
- **Hugging Face MiniLM-L6-v2** — embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- **Pinecone** — vector database for fast semantic search
- **Google Gemini 2.5 Pro** — primary LLM for generation
- **Flask** — lightweight web framework for serving the application
- **Docker** — containerization for reproducible deployments
- **AWS EC2** — production hosting environment
- **GitHub Actions** — end-to-end CI/CD pipeline (build, test, deploy)

---

## 🛠️ **How the System Works (Technical Overview)**

1. **Document Ingestion**  
   Stoic texts are parsed, cleaned, chunked, and embedded using a transformer-based embedding model.

2. **Vector Storage (Pinecone)**  
   All text chunks are stored as vectors, enabling high-recall semantic search.

3. **Query Processing**  
   When a user asks a question, the system:
   - Embeds the query
   - Retrieves the top-k relevant Stoic passages
   - Injects them into a **RAG prompt template**

4. **LLM Generation**  
   Gemini-2.5-Pro generates a response grounded in:
   - Retrieved Stoic text  
   - The system prompt (Stoic mentor persona)  
   - User’s question or reflection

5. **Flask Web Server**  
   Serves the chatbot UI and RAG pipeline via an API endpoint.

6. **Deployment & DevOps**  
   The entire application runs inside a Docker container, deployed automatically to an AWS EC2 instance using GitHub Actions.

---

## 📦 **Running Locally**

1. **Create a new Conda environment:**
   ```bash
   conda create -n stoicmind python=3.10
   conda activate stoicmind
