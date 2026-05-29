#step 1 - Read Transcript File

with open("/Users/omshelar/Desktop/course_RAG/RAG_vdeos/intro.txt", "r") as f:
    text = f.read()

print(text[:1000])


#Step 2 — Chunking

from sentence_transformers import SentenceTransformer

with open("/Users/omshelar/Desktop/course_RAG/RAG_vdeos/intro.txt", "r") as f:
    text = f.read()

chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])

print("Total chunks:", len(chunks))

#Step 3 - Embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

print(embeddings.shape)

#step 4 - Similarity Search
from sklearn.metrics.pairwise import cosine_similarity

query = "What is REPL?"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]

top_k = similarities.argsort()[-3:][::-1] #retrieve top 3 most similar chunks

print("\nTop Matching Chunks:\n")

for i in top_k:
    print(chunks[i])
    print("\n" + "="*80 + "\n")

#step 5 - LLM Prompting
import ollama

context = ""

for i in top_k:
    context += chunks[i] + "\n"

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

response = ollama.chat(
    model='llama3',
    messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ]
)

print("\nAI Answer:\n")
print(response['message']['content'])    