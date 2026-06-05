import streamlit as st
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Set page config for a premium, widescreen layout
st.set_page_config(
    page_title="Interactive Python Tutor & RAG Bot",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CONFIGURATION: Paste your YouTube tutorial video URL and paths here
# ==============================================================================
YOUTUBE_URL = "https://www.youtube.com/watch?v=rfscVS0vtbw"
TRANSCRIPT_PATH = "/Users/omshelar/Desktop/course_RAG/Transcripts/intro.txt"
MODEL_NAME = "llama3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ==============================================================================

# Custom CSS for modern premium dashboard aesthetics
st.markdown("""
<style>
    /* Import modern typography from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Apply styles globally */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Main background tuning */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* Gradient Typography for Main Title */
    .gradient-title {
        background: linear-gradient(135deg, #FFE873 0%, #306998 50%, #4B8BBE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    .subtitle-text {
        font-size: 1.15rem;
        color: #8b949e;
        margin-bottom: 25px;
        font-weight: 400;
    }


    /* Premium card panels */
    .premium-panel {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 105, 152, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Custom chatbot header styling */
    .chat-section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid rgba(88, 166, 255, 0.2);
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* Source citations styles */
    .source-citation {
        background-color: #161b22;
        border-left: 3px solid #ff7b72;
        border-radius: 4px;
        padding: 10px 15px;
        margin-top: 10px;
        font-size: 0.88rem;
        color: #8b949e;
    }

    .score-badge {
        display: inline-block;
        background-color: rgba(48, 105, 152, 0.3);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* Micro-interactions: Custom button animation */
    div.stButton > button {
        background: linear-gradient(135deg, #306998 0%, #1f4263 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
        background: linear-gradient(135deg, #4B8BBE 0%, #306998 100%);
        color: white;
    }

    /* Styled chat container */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load embedding model (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Helper function to read, chunk, and embed transcript (cached)
@st.cache_data
def load_and_preprocess_transcript(file_path):
    if not os.path.exists(file_path):
        return None, None
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Text chunking
    chunk_size = 500
    overlap = 100
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
        
    # Generate embeddings
    model = load_embedding_model()
    embeddings = model.encode(chunks)
    return chunks, embeddings

# Perform similarity search and return top matching chunks
def similarity_search(query, model, chunks, embeddings, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            "chunk": chunks[idx],
            "score": float(similarities[idx]),
            "idx": idx
        })
    return results

# Generate LLM response using Ollama
def generate_response(query, context):
    prompt = f"""You are a helpful AI assistant answering questions from a course transcript.

Rules:
1. Use ONLY the provided context.
2. If the answer exists in the context, answer directly.
3. Keep the answer concise, detailed and accurate.
4. If the answer is not available in the context, say:
   "I could not find the answer in the provided transcript."

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error connecting to local Ollama server: {str(e)}. Make sure Ollama is running (`ollama serve`) and model '{MODEL_NAME}' is pulled."

# Load RAG resources
with st.spinner("Initializing models and indexing tutorial transcript..."):
    embed_model = load_embedding_model()
    chunks, embeddings = load_and_preprocess_transcript(TRANSCRIPT_PATH)

# RAG default search parameters
min_similarity = 0.15
show_sources = True

# ==============================================================================
# MAIN PAGE SYSTEM
# ==============================================================================
st.markdown("<div class='gradient-title'>Python basics Live Stream Course</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Watch the Python Tutorial livestream and ask our AI RAG Bot about anything discussed in the class.</div>", unsafe_allow_html=True)

# Main container layout split into two columns: Video (Left) and chat info (Right)
col_left, col_right = st.columns([5, 3])

with col_left:
    st.markdown("### 🎥 Tutorial Video")
    # Streamlit video container playing youtube video
    st.video(YOUTUBE_URL)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # RAG BOT Section directly under the video player
    st.markdown("<div class='chat-section-header'>💬 Interactive RAG Assistant</div>", unsafe_allow_html=True)
    st.markdown("*Ask questions directly about what is discussed in the video above! (Powered by Sentence-Transformers & Llama3)*")
    
    # Initialize chat history session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and show_sources and msg["sources"]:
                with st.expander("📚 View Retrieved Transcript Segments"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-citation">
                            <span class="score-badge">Similarity Score: {src['score']:.4f}</span><br>
                            "{src['chunk']}"
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat Input Box
    if query := st.chat_input("Ask a question about the video (e.g. 'What is modulo?' or 'How old was Sam?')"):
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Generate answer using RAG
        with st.chat_message("assistant"):
            if chunks is None or embeddings is None:
                response = "Error: The transcript file `intro.txt` could not be found. Please check your transcript path."
                sources_to_store = []
                st.markdown(response)
            else:
                with st.spinner("Searching transcript & consulting Llama3..."):
                    # Find similar segments
                    search_results = similarity_search(query, embed_model, chunks, embeddings)
                    
                    # Filter results based on similarity threshold
                    filtered_results = [r for r in search_results if r["score"] >= min_similarity]
                    
                    if not filtered_results:
                        response = "I could not find the answer in the provided transcript (relevance similarity below threshold)."
                        sources_to_store = []
                        st.markdown(response)
                    else:
                        context = "\n\n".join([r["chunk"] for r in filtered_results])
                        response = generate_response(query, context)
                        sources_to_store = filtered_results
                        st.markdown(response)
                        
                        if show_sources and sources_to_store:
                            with st.expander("📚 View Retrieved Transcript Segments"):
                                for src in sources_to_store:
                                    st.markdown(f"""
                                    <div class="source-citation">
                                        <span class="score-badge">Similarity Score: {src['score']:.4f}</span><br>
                                        "{src['chunk']}"
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
            # Save assistant response to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources_to_store
            })

with col_right:
    # Sidebar style info cards for reference & notes
    st.markdown("### 💡 Quick Learning Notes")
    
    st.markdown("""
    <div class="premium-panel">
        <h4 style="color:#FFE873; margin-top:0;">⚡ The REPL / IDLE Shell</h4>
        <p style="font-size:0.95rem; line-height:1.4;">
            Python works as an interactive shell. REPL stands for <b>Read, Evaluate, Print, Loop</b>.
            It's like live television—you type code, hit Enter, and Python runs it instantly in memory!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="premium-panel">
        <h4 style="color:#FFE873; margin-top:0;">📦 The 4 Basic Variable Types</h4>
        <ul style="font-size:0.92rem; padding-left: 20px; line-height: 1.5; margin-bottom: 0;">
            <li><b>Integer (int)</b>: Whole numbers, e.g., <code>x = 5</code></li>
            <li><b>Float</b>: Decimal numbers, e.g., <code>y = 3.14</code></li>
            <li><b>String (str)</b>: Text enclosed in quotation marks, e.g., <code>name = "Sam"</code></li>
            <li><b>Boolean (bool)</b>: Binary state, <code>True</code> or <code>False</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-panel">
        <h4 style="color:#FFE873; margin-top:0;">⚙️ Handy Built-in Functions</h4>
        <ul style="font-size:0.92rem; padding-left: 20px; line-height: 1.5; margin-bottom: 0;">
            <li><code>type(obj)</code>: Tells you what class/type the object is.</li>
            <li><code>dir(obj)</code>: Lists all methods and attributes available for the object.</li>
            <li><code>help(obj)</code>: Displays documentation and details about a type or function.</li>
            <li><code>len(container)</code>: Returns the number of items in a list or characters in a string.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="premium-panel">
        <h4 style="color:#FFE873; margin-top:0;">🔄 Loop & Range</h4>
        <p style="font-size:0.92rem; line-height:1.4; margin-bottom:0;">
            <code>range(start, stop, step)</code> generates a sequence of integers. 
            Remember: <b>stop is exclusive</b> (up to, but not including).
        </p>
    </div>
    """, unsafe_allow_html=True)
