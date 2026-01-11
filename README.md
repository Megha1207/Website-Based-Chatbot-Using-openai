# AI Website Chatbot

## Project Overview

This project implements an AI-powered chatbot that answers questions **strictly based on the content of a user-provided website**. The system crawls a website, extracts meaningful textual content, converts it into embeddings, stores those embeddings persistently, and enables grounded question answering through semantic retrieval.

The chatbot is explicitly designed to **avoid hallucinations**. If a question cannot be answered using the website content, it responds exactly with:

> **"The answer is not available on the provided website."**

The solution follows a retrieval-augmented generation (RAG) architecture with multiple safeguards to ensure correctness, relevance, and grounding.

---

## Architecture Explanation

The system is divided into **two main phases**:

### 1. Website Indexing Pipeline

This phase runs once per website (or during re-indexing):

1. **URL Validation** - Ensures the input URL is valid and reachable; handles empty, blocked, or unsupported websites gracefully
2. **Website Crawling** - Crawls HTML pages starting from the provided URL; avoids duplicates; respects domain boundaries
3. **Content Extraction** - Extracts meaningful textual content only; removes headers, footers, navigation menus, and advertisements
4. **Text Processing and Chunking** - Cleans and normalizes text; splits into overlapping semantic chunks with configurable size and overlap; retains metadata (source URL, page title, crawl depth)
5. **Embedding Generation** - Converts each text chunk into a dense vector embedding for reuse
6. **Vector Storage** - Stores embeddings persistently in a vector database; each website has its own isolated vector store

### 2. Question Answering Pipeline

This phase runs for every user query:

1. **Query Embedding** - The user question is embedded using the same embedding model
2. **Semantic Retrieval** - Relevant chunks are retrieved using cosine similarity with distance thresholds
3. **Context Assembly** - Multiple relevant chunks are combined to form site-wide context
4. **Grounding Validation** - Ensures retrieved context supports the question; returns fallback if not
5. **LLM Answer Generation** - A local LLM generates answers using only retrieved context
6. **Post-Generation Validation** - Validates that answers don't introduce external facts, unsupported entities, or numeric hallucinations

---

## Frameworks & Technologies

- **LangChain / LangGraph**: Not used - Project avoids heavy orchestration frameworks to keep logic explicit, auditable, and transparent
- **LLM Model**: TinyLLaMA (via Ollama) - Fully local inference with deterministic behavior; suitable for context-grounded QA
- **Vector Database**: ChromaDB (persistent mode) - Lightweight, supports cosine similarity search, persistent storage, per-website isolation
- **Embedding Model**: SentenceTransformers - Converts semantic chunks into dense vectors; uses cosine similarity for retrieval

---

## Setup Instructions

### 1. Create a virtual environment
```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama
```bash
ollama pull tinyllama
```

Ensure the Ollama service is running.

### 5. Run the Streamlit application
```bash
streamlit run app.py
```

---

## User Interface

The Streamlit interface allows users to:
- Enter a website URL
- Index or re-index the website
- Ask questions via a chat interface
- View conversation history
- Receive clear fallback responses when answers are unavailable

Re-indexing is automatically skipped if embeddings exist, unless explicitly requested.

---

## Assumptions & Limitations

### Assumptions
- Websites primarily contain HTML-based textual content
- JavaScript-rendered content may be partially unsupported
- Answer quality depends on extracted text quality
- Designed for factual, informational websites

### Limitations
- No support for PDFs or image-based text
- Crawling speed depends on website size and network conditions
- Local LLM performance depends on available system memory
- No multilingual support in current version

---

## Future Improvements

- Asynchronous crawling for faster indexing
- Improved boilerplate removal for complex websites
- Source citation display for each answer
- Docker-based deployment
- FastAPI backend for scalability
- Optional hybrid keyword + semantic retrieval
- Advanced re-ranking models for improved precision

---



## Deployment (Local – Ollama)

This project supports fully local deployment using Ollama for Large Language Model inference. This mode requires no external API keys and ensures all inference runs entirely on the local machine.

### Prerequisites

- Python 3.10+
- Sufficient system memory (minimum 6 GB recommended)
- Ollama installed and running

### Install Ollama

Download and install Ollama from:

[https://ollama.com](https://ollama.com)

Verify installation:

```bash
ollama --version
```

### Pull the LLM Model

The project is configured to use a lightweight local model by default.

```bash
ollama pull tinyllama
```

Ensure the model is available:

```bash
ollama list
```

### Activate Virtual Environment

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start Ollama Service

Ensure Ollama is running in the background:

```bash
ollama run tinyllama
```

You may exit the prompt after confirming it starts successfully.

### Run the Application
```bash
streamlit run app.py
```

The application will automatically detect and use Ollama as the LLM provider when no cloud API key is configured.

### Notes

- All LLM inference runs locally on the user’s machine
- No data is sent to external services
- Performance depends on available system memory and CPU
- Larger models may require additional RAM

### When to Use Ollama Deployment

- Local development and testing
- Privacy-sensitive data
- Offline environments
- Demonstrating system design without cloud dependencies