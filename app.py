import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

st.title("Поиск по коду")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

try:
    collection = load_collection()
except Exception:
    st.error("База не найдена. Сначала запусти: python index.py")
    st.stop()

query = st.text_input("Введи вопрос о коде (на русском или английском)")
search = st.button("Найти")


if search and query:
    results = collection.query(query_texts=[query], n_results=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        relevance = round((1 - dist) * 100)
        st.markdown(f"**#{i+1} — {meta.get('name')}** | 📁 `{meta.get('path')}` | Релевантность: **{relevance}%**")
        st.code(doc, language="python")
        st.divider()
