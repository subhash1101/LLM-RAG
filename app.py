import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ---- Load Local Model ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

tokenizer, model = load_model()

# ---- Load Knowledge Base into ChromaDB ----
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_knowledge_base():
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    if collection.count() == 0:
        for i, doc in enumerate(tqdm(docs, desc="Indexing knowledge base")):
            collection.add(documents=[doc], ids=[f"doc_{i}"], embeddings=sentence_transformer_ef([doc]))
load_knowledge_base()

# ---- RAG Search ----
def search_knowledge(query, top_k=3):
    query_embedding = sentence_transformer_ef([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# ---- Generate Response ----
def generate_response(user_query, context_docs):
    context_text = "\n".join(f"- {doc}" for doc in context_docs)
    prompt = f"""
You are a helpful assistant. Use the following knowledge to answer the question.
If the knowledge is not useful, answer using your general knowledge.

Knowledge:
{context_text}

Question:
{user_query}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ---- UI Setup ----
# st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🧠", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.markdown("Ask anything from your custom knowledge base.")
    st.markdown("📂 `knowledge_base.txt` is the source file.")
    st.markdown("💡 Answers are based on both context and general AI knowledge.")
    st.markdown("---")
    st.markdown("👨‍💻 Created using:")
    st.markdown("- `Streamlit` for UI\n- `ChromaDB` for vector search\n- `Flan-T5-Large` locally for LLM")
    st.markdown("📎 Tip: Keep questions factual for best results.")

# ---- Header ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 RAG Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions using your custom knowledge base + AI generation</p>", unsafe_allow_html=True)

# ---- Chat History ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Layout: Input and Output Side-by-Side ----
col1, col2 = st.columns([2, 3])

with col1:
    user_input = st.text_input("💬 Ask your question here:")
    if user_input:
        with st.spinner("Searching and thinking..."):
            context = search_knowledge(user_input)
            answer = generate_response(user_input, context)
        st.session_state.chat_history.append(("🧑 You", user_input))
        st.session_state.chat_history.append(("🤖 Assistant", answer))

with col2:
    st.markdown("### 💬 Chat History")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <strong style="color:#333;">{speaker}</strong><br>
            <span style="color:#222;">{msg}</span>
        </div>
        """, unsafe_allow_html=True)
