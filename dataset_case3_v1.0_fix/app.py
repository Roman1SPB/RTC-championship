import json
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from llm import get_llm_explanation
from score import score_question

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "code_embs"
EVAL_PATH = "eval_questions.json"
TARGET = 0.60

st.set_page_config(page_title="CodeLens", page_icon="🔎", layout="wide")


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    # корпус для BM25 (гибридный поиск)
    data = collection.get(include=["documents", "metadatas"])
    all_ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    id2doc = {i: (d, m) for i, d, m in zip(all_ids, docs, metas)}
    return collection, model, bm25, all_ids, id2doc


def chunk_id_of(meta: dict) -> str:
    return meta.get("chunk_id") or meta.get("info") or ""


def hybrid_search(query, collection, model, bm25, all_ids, alpha=0.7, top_k=5, n_candidates=30):
    """Комбинация векторного поиска и BM25 с настраиваемым весом alpha.
       alpha=1.0 — только семантика; alpha=0.0 — только точные слова."""
    vec = model.encode(query).tolist()
    vec_res = collection.query(query_embeddings=[vec], n_results=n_candidates)
    vec_scores = {i: 1 - d for i, d in zip(vec_res["ids"][0], vec_res["distances"][0])}

    bm25_raw = np.array(bm25.get_scores(tokenize(query)))
    mx = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / mx
    top = np.argsort(bm25_raw)[::-1][:n_candidates]
    bm25_scores = {all_ids[i]: float(bm25_norm[i]) for i in top if bm25_norm[i] > 0}

    ids = set(vec_scores) | set(bm25_scores)
    final = {i: alpha * vec_scores.get(i, 0.0) + (1 - alpha) * bm25_scores.get(i, 0.0) for i in ids}
    return sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ─────────────────────────── Страница: Поиск + чат ───────────────────────────
def page_search():
    st.title("🔎 Поиск по коду")
    try:
        collection, model, bm25, all_ids, id2doc = load_resources()
    except Exception:
        st.error("База не найдена. Сначала запусти: `python index.py gymhero`")
        st.stop()

    st.sidebar.header("Настройки поиска")
    alpha = st.sidebar.slider("Вес векторного поиска (α)", 0.0, 1.0, 0.7, 0.05)
    st.sidebar.caption("α=1.0 — только семантика · α=0.0 — только точные слова (BM25)")

    if "last_search" not in st.session_state:
        st.session_state.last_search = None

    with st.expander("Поиск по коду (без объяснений LLM)", expanded=True):
        q = st.text_input("Введите запрос", key="search_input_unique")
        if st.button("Найти фрагменты", key="search_btn_unique") and q:
            hits = hybrid_search(q, collection, model, bm25, all_ids, alpha=alpha, top_k=5)
            docs, metas, rels = [], [], []
            for cid, score in hits:
                doc, meta = id2doc.get(cid, ("", {"info": cid}))
                docs.append(doc); metas.append(meta); rels.append(score)
            st.session_state.last_search = (docs, metas, rels)

            for doc, meta, score in zip(docs, metas, rels):
                info = chunk_id_of(meta)
                lang = "java" if ".java" in info else "python"
                st.markdown(f"**{info}** · релевантность **{round(score * 100)}%**")
                st.code(doc, language=lang)
                st.divider()

    # чат с LLM
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Введи вопрос о коде"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if st.session_state.last_search is None:
            st.warning("Сначала найдите фрагменты кода через блок поиска.")
            st.stop()
        documents, metadatas, _ = st.session_state.last_search
        code_chunks = documents[:3]
        chunk_names = [chunk_id_of(metadatas[i]) for i in range(min(3, len(metadatas)))]
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
        collection, model, bm25, all_ids, id2doc = load_resources()
    except Exception:
        st.error("База не найдена. Сначала запусти: `python index.py gymhero`")
        st.stop()

    questions = json.loads(Path(EVAL_PATH).read_text(encoding="utf-8"))

    use_hybrid = st.checkbox("Гибридный поиск (вектор + BM25)", value=False)
    alpha = st.slider("Вес векторного поиска (α)", 0.0, 1.0, 0.7, 0.05) if use_hybrid else 1.0

    if st.button("▶ Посчитать Precision@5", type="primary", use_container_width=True):
        rows, latencies, scores = [], [], []
        with st.spinner("Прогоняю 15 вопросов…"):
            for q in questions:
                t0 = time.perf_counter()
                if use_hybrid:
                    hits = hybrid_search(q["query"], collection, model, bm25, all_ids, alpha=alpha, top_k=5)
                    top5 = [cid for cid, _ in hits]
                else:
                    vec = model.encode(q["query"]).tolist()
                    res = collection.query(query_embeddings=[vec], n_results=5)
                    top5 = [chunk_id_of(m) for m in res["metadatas"][0]]
                latencies.append(time.perf_counter() - t0)
                s = score_question(top5, q["correct_chunk_ids"])
                scores.append(s)
                rows.append({"вопрос": q["question_id"], "язык": q.get("language"),
                             "сложность": q.get("difficulty"),
                             "эталонов": len(q["correct_chunk_ids"]), "score": s})

        mean = sum(scores) / len(scores)
        if mean >= TARGET:
            st.success(f"Precision@5 = {mean:.3f} — цель ≥ {TARGET:.0%} достигнута")
        else:
            st.warning(f"Precision@5 = {mean:.3f} — ниже цели {TARGET:.0%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@5", f"{mean:.3f}", delta=f"{mean - TARGET:+.3f} к цели")
        c2.metric("Средняя задержка", f"{sum(latencies) / len(latencies):.2f} с")
        c3.metric("Макс. задержка", f"{max(latencies):.2f} с")
        st.divider()

        def agg(key):
            d = {}
            for q, s in zip(questions, scores):
                d.setdefault(q.get(key), []).append(s)
            return {k: sum(v) / len(v) for k, v in d.items()}

        cA, cB = st.columns(2)
        with cA:
            st.subheader("По сложности")
            order = {"easy": 0, "medium": 1, "hard": 2}
            diff = dict(sorted(agg("difficulty").items(), key=lambda x: order.get(x[0], 9)))
            st.bar_chart(diff, horizontal=True, color="#7C3AED")
        with cB:
            st.subheader("По языку")
            st.bar_chart(agg("language"), horizontal=True, color="#F97316")

        st.divider()
        st.subheader("Детально по вопросам")
        st.dataframe(rows, hide_index=True, use_container_width=True,
                     column_config={"score": st.column_config.ProgressColumn(
                         "score", min_value=0.0, max_value=1.0, format="%.2f")})
    else:
        st.info("Нажми кнопку, чтобы прогнать все 15 вопросов.")


pg = st.navigation([
    st.Page(page_search, title="Поиск", icon="🔍", default=True),
    st.Page(page_metrics, title="Метрики", icon="📊"),
])
pg.run()
