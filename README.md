# 🎓 Tutorial Videos RAG System

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions from tutorial video transcripts using semantic search, embeddings, and local Large Language Models (LLMs).

Built using **Whisper**, **Sentence Transformers**, **Streamlit**, **Ollama**, and **Llama 3**.

---

## 🚀 Features

* 🎥 Embedded tutorial video interface
* 📝 Transcript-based question answering
* 🔍 Semantic search using Sentence Transformers
* 📚 Context retrieval from transcript chunks
* 🤖 Local LLM inference with Ollama (Llama 3)
* 💬 Interactive Streamlit chatbot interface
* 📖 Source citation display with similarity scores
* 🔄 Persistent chat history during session
* 🖥️ Fully local execution (No OpenAI API required)

---

## 🏗️ System Architecture

```text
Tutorial Video
      │
      ▼
Whisper Transcription
      │
      ▼
Transcript (.txt)
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
Semantic Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Llama 3 (Ollama)
      │
      ▼
AI Generated Answer
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### AI / NLP

* Sentence Transformers
* Ollama
* Llama 3
* Whisper

### Machine Learning

* Scikit-learn
* Cosine Similarity

### Programming Language

* Python

---

## 📂 Project Structure

```text
tutorial-videos-rag/
│
├── Transcripts/
│   └── intro.txt
│
├── app.py
├── RAG.py
├── README.md
├── requirements.txt
└── screenshots/
```

---

## ⚙️ How It Works

### Step 1: Video Transcription

The tutorial video is converted into text using Whisper.

### Step 2: Chunking

The transcript is divided into overlapping chunks.

```python
chunk_size = 500
overlap = 100
```

This preserves context between chunks.

### Step 3: Embedding Generation

Each chunk is converted into a numerical vector using:

```python
all-MiniLM-L6-v2
```

from Sentence Transformers.

### Step 4: Semantic Search

When the user asks a question:

1. The question is embedded.
2. Cosine similarity is calculated.
3. Top relevant chunks are retrieved.

### Step 5: Response Generation

Retrieved chunks are passed to:

```text
Llama 3 (Ollama)
```

which generates the final answer based only on the retrieved transcript context.

---

## 📸 Application Preview

### Main Dashboard

* Embedded tutorial video
* AI-powered chatbot
* Source citations
* Similarity scores

(Add screenshots here)

---

## ▶️ Running the Project

### Clone Repository

```bash
git clone https://github.com/OmShelar2004/tutorial-videos-rag.git

cd tutorial-videos-rag
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama

Download Ollama:

https://ollama.com

Pull Llama 3:

```bash
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Example Questions

* What is REPL?
* What are Python loops?
* How do variables work in Python?
* What is the difference between a list and tuple?
* What built-in functions were discussed?

---

## 📈 Future Improvements

* FAISS Vector Database Integration
* Multi-Video Support
* PDF RAG Support
* Hybrid Search (BM25 + Embeddings)
* Conversation Memory
* Deployment on Streamlit Cloud
* Multi-Document Knowledge Base

---

## 👨‍💻 Author

**Om Dilip Shelar**

LinkedIn:
https://www.linkedin.com/in/om-shelar04

GitHub:
https://github.com/OmShelar2004

---

## ⭐ Project Goal

This project was built as a hands-on implementation of Retrieval-Augmented Generation (RAG) concepts to understand semantic search, embeddings, vector retrieval, and local LLM-based question answering systems.
