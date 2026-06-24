import json
import time
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from llm import get_llm_explanation
from score import score_question

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"
EVAL_PATH = "eval_questions.json"
TARGET = 0.60

st.set_page_config(page_title="CodeLens", page_icon="🔎", layout="wide")


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model


# ─────────────────────────── Страница: Поиск + чат ───────────────────────────
def page_search():
    st.title("🔎 Поиск по коду")

    try:
        collection, model = load_resources()
    except Exception:
        st.error("База не найдена. Сначала запусти: `python index.py gymhero`")
        st.stop()

    if "last_search" not in st.session_state:
        st.session_state.last_search = None

    with st.expander("Поиск по коду (без объяснений LLM)", expanded=False):
        search_query = st.text_input("Введите запрос для поиска фрагментов", key="search_input_unique")
        if st.button("Найти фрагменты", key="search_btn_unique"):
            if search_query:
                vec = model.encode(search_query).tolist()
                res = collection.query(query_embeddings=[vec], n_results=5)
                st.session_state.last_search = (
                    res["documents"][0], res["metadatas"][0], res["distances"][0]
                )
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                    st.code(doc, language="python")
                    st.caption(f"Релевантность: {round((1 - dist) * 100)}% | {meta.get('info', '')}")
                    st.divider()

    # чат с LLM
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Введи вопрос о коде"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.last_search is None:
            st.warning("Сначала найдите фрагменты кода через верхний блок поиска.")
            st.stop()

        documents, metadatas, distances = st.session_state.last_search
        top_n = 3
        code_chunks = documents[:top_n]
        chunk_names = [metadatas[i].get("info", "Без имени") for i in range(top_n)]

        with st.spinner("Генерирую объяснение..."):
            explanation = get_llm_explanation(prompt, code_chunks, chunk_names)

        st.session_state.messages.append({"role": "assistant", "content": explanation})
        with st.chat_message("assistant"):
            st.markdown(explanation)


# ─────────────────────────── Страница: Метрики ───────────────────────────
def page_metrics():
    st.title("📊 Качество поиска — Precision@5")
    st.caption("Метрика по eval_questions.json, официальная логика score.py (допуск ±2 строки).")

    try:
        collection, model = load_resources()
    except Exception:
        st.error("База не найдена. Сначала запусти: `python index.py gymhero`")
        st.stop()

    questions = json.loads(Path(EVAL_PATH).read_text(encoding="utf-8"))

    if st.button("▶ Посчитать Precision@5", type="primary", use_container_width=True):
        rows, latencies, scores = [], [], []

        with st.spinner("Прогоняю 15 вопросов через поиск…"):
            for q in questions:
                t0 = time.perf_counter()
                vec = model.encode(q["query"]).tolist()
                res = collection.query(query_embeddings=[vec], n_results=5)
                latencies.append(time.perf_counter() - t0)

                top5 = [m.get("chunk_id") or m.get("info") for m in res["metadatas"][0]]
                s = score_question(top5, q["correct_chunk_ids"])
                scores.append(s)
                rows.append({
                    "вопрос": q["question_id"],
                    "язык": q.get("language"),
                    "сложность": q.get("difficulty"),
                    "эталонов": len(q["correct_chunk_ids"]),
                    "score": s,
                })

        mean = sum(scores) / len(scores)

        if mean >= TARGET:
            st.success(f"Precision@5 = {mean:.3f}  —  цель ≥ {TARGET:.0%} достигнута")
        else:
            st.warning(f"Precision@5 = {mean:.3f}  —  ниже цели {TARGET:.0%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@5", f"{mean:.3f}", delta=f"{mean - TARGET:+.3f} к цели")
        c2.metric("Средняя задержка", f"{sum(latencies) / len(latencies):.2f} с")
        c3.metric("Макс. задержка", f"{max(latencies):.2f} с")

        st.divider()

        def agg(key: str) -> dict:
            d: dict = {}
            for q, s in zip(questions, scores):
                d.setdefault(q.get(key), []).append(s)
            return {k: sum(v) / len(v) for k, v in d.items()}

        colA, colB = st.columns(2)
        with colA:
            st.subheader("По сложности")
            order = {"easy": 0, "medium": 1, "hard": 2}
            diff = dict(sorted(agg("difficulty").items(), key=lambda x: order.get(x[0], 9)))
            st.bar_chart(diff, horizontal=True, color="#7C3AED")
        with colB:
            st.subheader("По языку")
            st.bar_chart(agg("language"), horizontal=True, color="#F97316")

        st.divider()
        st.subheader("Детально по вопросам")
        st.dataframe(
            rows, hide_index=True, use_container_width=True,
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
    else:
        st.info("Нажми кнопку, чтобы прогнать все 15 вопросов и увидеть метрику.")


# ─────────────────────────── Навигация ───────────────────────────
pg = st.navigation([
    st.Page(page_search, title="Поиск", icon="🔍", default=True),
    st.Page(page_metrics, title="Метрики", icon="📊"),
])
pg.run()
