import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llm import get_llm_explanation

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

st.title("Поиск по коду") #отображает заголовок

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

    # Фрагменты уже показаны сверху, в чате их не выводим (убрал expander)
    top_n = 3 #подготовка контекста для llm
    code_chunks = documents[:top_n]
    chunk_names = [metadatas[i].get('info', 'Без имени') for i in range(top_n)]

    with st.spinner("Генерирую объяснение..."):#генерация объяснения
        explanation = get_llm_explanation(prompt, code_chunks, chunk_names)

    st.session_state.messages.append({"role": "assistant", "content": explanation})#добавление ответа в историю и отображение
    with st.chat_message("assistant"):
        st.markdown(explanation)