import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ---- Initialize ChromaDB ----
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge")

# ---- Use Sentence Transformers for Embeddings ----
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ---- Load Hugging Face Model Locally ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

tokenizer, model = load_model()

# ---- Load Knowledge Base and Store in ChromaDB ----
@st.cache_resource
def load_knowledge_base():
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    existing = collection.count()
    if existing == 0:
        for i, doc in enumerate(tqdm(docs, desc="Indexing knowledge base")):
            collection.add(
                documents=[doc],
                ids=[f"doc_{i}"],
                embeddings=sentence_transformer_ef([doc])
            )

load_knowledge_base()

# ---- RAG Search Function ----
def search_knowledge(query, top_k=3):
    query_embedding = sentence_transformer_ef([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# ---- Generate Response Using Local Model ----
def generate_response(user_query, context_docs):
    context_text = "\n".join(f"- {doc}" for doc in context_docs)
    
    prompt = f"""
You are a helpful assistant. Answer the following question .

Knowledge:
{context_text}

Question:
{user_query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# ---- Streamlit Chat UI ----
st.title("🔍 RAG-Powered Knowledge Assistant (Local Model)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        context = search_knowledge(user_input)
        answer = generate_response(user_input, context)
    
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", answer))

for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

