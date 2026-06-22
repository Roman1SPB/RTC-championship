import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

st.title("Поиск по коду")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model

try:
    collection, model = load_resources()
except Exception as e:
    st.error("База не найдена. Сначала запусти: python index.py")
    st.stop()

query = st.text_input("Введи вопрос о коде (на русском или английском)")
search = st.button("Найти")

if search and query:
    query_vector = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        relevance = round((1 - dist) * 100)
        st.markdown(f"**#{i+1} — {meta.get('info')}** | Релевантность: **{relevance}%**")
        st.code(doc, language="python")
        st.divider()