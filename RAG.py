
# STEP 1 - READ TRANSCRIPT


with open("/Users/omshelar/Desktop/course_RAG/Transcripts/intro.txt", "r") as f:
    text = f.read()


# STEP 2 - CHUNKING


from sentence_transformers import SentenceTransformer

chunk_size = 500
overlap = 100

chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunks.append(text[i:i + chunk_size])

print(f"\nTotal chunks: {len(chunks)}")


# STEP 3 - EMBEDDINGS


model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating embeddings...")

embeddings = model.encode(chunks)

print("Embeddings shape:", embeddings.shape)


# STEP 4 - IMPORTS


from sklearn.metrics.pairwise import cosine_similarity
import ollama


# STEP 5 - CHAT LOOP


while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("\nGoodbye 👋")
        break

   
    # QUERY EMBEDDING
    

    query_embedding = model.encode([query])

   
    # SIMILARITY SEARCH
   

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    
    # TOP 5 CHUNKS
    

    top_k = similarities.argsort()[-5:][::-1]

    print("\nTop Retrieved Chunks:\n")

    context_parts = []

    for rank, idx in enumerate(top_k, start=1):

        print(f"Chunk {rank}")
        print(f"Similarity Score: {similarities[idx]:.4f}")

        preview = chunks[idx][:250]
        print(preview)
        print("-" * 60)

        context_parts.append(chunks[idx])

    
    # BUILD CONTEXT
    

    context = "\n\n".join(context_parts)

    
    # PROMPT
    

    prompt = f"""
You are a helpful AI assistant answering questions from a course transcript.

Rules:
1. Use ONLY the provided context.
2. If the answer exists in the context, answer directly.
3. Keep the answer concise and accurate.
4. If the answer is not available in the context, say:
   "I could not find the answer in the provided transcript."

Context:
{context}

Question:
{query}

Answer:
"""

    
    # LLM CALL
   

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

   
    # FINAL ANSWER
    

    print("\n" + "=" * 60)
    print("AI Answer:\n")
    print(response["message"]["content"])
    print("=" * 60)