# Tutorial Videos RAG System

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions from tutorial video transcripts using semantic search, embeddings, and local Large Language Models (LLMs).

## Features

* Video-to-text transcription using Whisper
* Transcript preprocessing and chunking
* Semantic embeddings using Sentence Transformers
* Cosine similarity-based semantic retrieval
* Top-K context retrieval
* Local LLM inference using Ollama and Llama 3
* Context-aware AI-generated responses

---

# Project Workflow

```text
Tutorial Videos
       ↓
Whisper Transcription
       ↓
Transcript Chunking
       ↓
Embedding Generation
       ↓
Semantic Search
       ↓
Top-K Retrieval
       ↓
LLM Response Generation
```

---

# Technologies Used

* Python
* Whisper
* Sentence Transformers
* Scikit-learn
* Ollama
* Llama 3
* FFmpeg

---

# How It Works

1. Tutorial videos are transcribed into text using Whisper.
2. The transcript is divided into smaller chunks.
3. Sentence embeddings are generated for each chunk.
4. User queries are converted into embeddings.
5. Cosine similarity is used to retrieve the most relevant chunks.
6. Retrieved context is passed to a local LLM for answer generation.

---

# Example Query

```python
query = "What are Python loops?"
```

Example Output:

```text
Python loops are used for repeatable execution of code blocks...
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/tutorial-videos-rag.git
```

## Install Dependencies

```bash
pip install sentence-transformers scikit-learn ollama
```

## Run Project

```bash
python3 RAG.py
```

---

# Future Improvements

* Streamlit UI
* ChromaDB integration
* Conversational memory
* Timestamp-based retrieval
* Multi-video support
* Hybrid search
* Reranking

---

# Learning Outcomes

This project helped in understanding:

* Retrieval-Augmented Generation (RAG)
* Semantic embeddings
* Vector similarity search
* Context-aware LLM prompting
* AI pipeline architecture
* Local LLM workflows

---

# Author

Om Dilip Shelar
