import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llm import get_llm_explanation

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

    st.subheader("Найденные фрагменты кода")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        relevance = round((1 - dist) * 100)
        name = meta.get('info', 'Без имени')
        st.markdown(f"**#{i+1} — {name}** | Релевантность: **{relevance}%**")
        if ("java" in name):
            st.code(doc, language="java")
        if ("py" in name):
            st.code(doc, language="python")
        
        st.divider()
    top_n = 5
    code_chunks = documents[:top_n]
    chunk_names = [
        (metadatas[i].get('info', 'Без имени'))
        for i in range(top_n)
    ]

    with st.spinner("Генерирую объяснение..."):
        explanation = get_llm_explanation(query, code_chunks, chunk_names)

    st.subheader("Объяснение от AI")
    st.write(explanation)