import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llm import get_llm_explanation
from rank_bm25 import BM25Okapi
import re
import numpy as np

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

st.title("Поиск по коду") #отображает заголовок

st.sidebar.header("Настройки гибридного поиска")
alpha = st.sidebar.slider("Вес векторного поиска (α)", 0.0, 1.0, 0.7, 0.05)
st.sidebar.caption("α = 1.0: только семантика\nα = 0.0: только точные слова (BM25)")

def tokenize(text):
    return re.findall(r'\w+', text.lower())

@st.cache_resource # кеширует результат работы функции load_resources()
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME) #берем фрагменты кода code_embs
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model
try:
    collection, model = load_resources()
except Exception as e:
    st.error("Сначала запусти: python index.py")
    st.stop()

def hybrid_search(query, alpha=0.7, top_k=5, n_candidates=30):
    """Гибридный поиск: векторный + BM25"""
    vec = model.encode(query).tolist()
    vec_res = collection.query(query_embeddings=[vec], n_results=n_candidates)
    vec_scores = {id_: 1 - dist for id_, dist in zip(vec_res['ids'][0], vec_res['distances'][0])}
    
    bm25_raw = bm25.get_scores(tokenize(query))
    max_bm25 = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / max_bm25
    top_idx = np.argsort(bm25_raw)[::-1][:n_candidates]
    bm25_scores = {all_ids[i]: bm25_norm[i] for i in top_idx if bm25_norm[i] > 0}
    
    all_ids_set = set(vec_scores.keys()) | set(bm25_scores.keys())
    final = {id_: alpha * vec_scores.get(id_, 0.0) + (1 - alpha) * bm25_scores.get(id_, 0.0) for id_ in all_ids_set}
    
    return sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]

# --- хранилище последних найденных фрагментов ---
if "last_search" not in st.session_state:
    st.session_state.last_search = None

# блок отдельного поиска
with st.expander("Поиск по коду (без объяснений LLM)", expanded=False): #создаем раскрывающийся блок
    search_query = st.text_input("Введите запрос для поиска фрагментов", key="search_input_unique") #поле ввода запроса
    if st.button("Найти фрагменты", key="search_btn_unique"): #кнопка запуска поиска.Ключ нужен чтобы не код не перепутал,какая кнопка нажата
        if search_query: #если запрос не пустой
            vec = model.encode(search_query).tolist() #превращаем его в вектор
            res = collection.query(query_embeddings=[vec], n_results=3) #ищем похожие фрагменты в базе
            # сохраняем результаты в session_state
            st.session_state.last_search = (
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0]
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                st.code(doc, language="python")
                st.caption(f"Релевантность: {round((1-dist)*100)}% | {meta.get('info', '')}")
                st.divider() #выводим все результаты

# чат с LLM
if "messages" not in st.session_state: #инициализация истории чата
    st.session_state.messages = []

for message in st.session_state.messages: #отображение всех предыдущих сообщений
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Введи вопрос о коде"):#ждем ввод от пользователя.Текст попадает в переменную prompt
    st.session_state.messages.append({"role": "user", "content": prompt}) #Добавление вопроса пользователя в историю и его отображение
    with st.chat_message("user"):
        st.markdown(prompt)

    # используем сохранённые результаты, а не новый поиск
    if st.session_state.last_search is None:
        st.warning("Сначала найдите фрагменты кода через верхний блок поиска.")
        st.stop()

    documents, metadatas, distances = st.session_state.last_search

    top_n = 3 #подготовка контекста для llm
    code_chunks = documents[:top_n]
    chunk_names = [metadatas[i].get('info', 'Без имени') for i in range(top_n)]

    with st.spinner("Генерирую объяснение..."):#генерация объяснения
        explanation = get_llm_explanation(prompt, code_chunks, chunk_names)

    st.session_state.messages.append({"role": "assistant", "content": explanation})#добавление ответа в историю и отображение
    with st.chat_message("assistant"):
        st.markdown(explanation)