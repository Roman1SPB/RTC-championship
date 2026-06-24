"""
Страница метрик — Precision@5.

Прогоняет 15 вопросов из eval_questions.json через тот же поиск (ChromaDB + модель)
и считает метрику ОФИЦИАЛЬНОЙ функцией score_question из score.py.
Дополнительно: средняя/макс задержка, разбивки по языку и сложности, таблица по вопросам.

Запуск: python -m streamlit run app.py  (страница появится в сайдбаре)
"""

import json
import time
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from score import score_question

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"
EVAL_PATH = "eval_questions.json"
TARGET = 0.60

st.set_page_config(page_title="Метрики — Precision@5", layout="wide")
st.title("Качество поиска — Precision@5")
st.caption("Метрика по eval_questions.json, официальная логика score.py (допуск ±2 строки).")


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model


try:
    collection, model = load_resources()
except Exception:
    st.error("База не найдена. Сначала запусти: `python index.py gymhero`")
    st.stop()

questions = json.loads(Path(EVAL_PATH).read_text(encoding="utf-8"))

run = st.button("▶ Посчитать Precision@5", type="primary", use_container_width=True)

if run:
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

    # --- крупная метрика + статус ---
    if mean >= TARGET:
        st.success(f"Precision@5 = {mean:.3f}  —  цель ≥ {TARGET:.0%} достигнута")
    else:
        st.warning(f"Precision@5 = {mean:.3f}  —  ниже цели {TARGET:.0%}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@5", f"{mean:.3f}", delta=f"{mean - TARGET:+.3f} к цели")
    c2.metric("Средняя задержка", f"{sum(latencies) / len(latencies):.2f} с")
    c3.metric("Макс. задержка", f"{max(latencies):.2f} с")

    st.divider()

    # --- разбивки в виде столбчатых диаграмм ---
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

    # --- таблица по вопросам с цветовой шкалой score ---
    st.subheader("Детально по вопросам")
    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
else:
    st.info("Нажми кнопку, чтобы прогнать все 15 вопросов и увидеть метрику.")
