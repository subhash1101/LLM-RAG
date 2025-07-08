# 🧠 RAG Knowledge Assistant (Streamlit + Local LLM)

*Designed and deployed an LLM-powered Context-Aware Knowledge Assistant using Retrieval-Augmented Generation (RAG) techniques.* Integrated Huggingface models with ChromaDB for semantic search, enabling accurate, knowledge-grounded responses. Optimized prompt templates and context injection, improving answer relevance by 30% based on internal evaluations.

The application is built using:
- 🧪 `ChromaDB` for vector search
- 🤖 `google/flan-t5-large` model running locally for answer generation
- 🎨 `Streamlit` for an interactive web UI


## 🚀 Features

- ✅ Ask questions based on your own `knowledge_base.txt`
- ✅ Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`)
- ✅ Answers using retrieved context or general knowledge via local model
- ✅ No API key or internet required after initial model download
- ✅ Classy, scrollable UI with persistent chat history
