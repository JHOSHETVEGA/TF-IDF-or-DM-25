import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from preprocess import load_spacy, get_stopwords, basic_clean, tokenize_lemmatize
from retrieval import TfidfSearcher, BM25Searcher

# ---------- Config ----------
st.set_page_config(page_title="IR 20NG: TF-IDF vs BM25", page_icon="🔎", layout="centered")

# ---------- Datos ----------
@st.cache_data(show_spinner=False)
def load_docs():
    cats = ["sci.space", "rec.sport.baseball", "talk.politics.misc"]
    data = fetch_20newsgroups(
        subset="train",
        categories=cats,
        remove=("headers", "footers", "quotes"),
    )
    docs = data.data[:200]   # subset de 200 documentos
    ids  = [f"20NG_{i}" for i in range(len(docs))]
    return docs, ids

# ---------- Índices (ambos modelos) ----------
@st.cache_resource(show_spinner=False)
def build_indices(docs, ids):
    nlp = load_spacy("en")
    stop_set = get_stopwords("en")
    cleaned = [basic_clean(d) for d in docs]
    tokenized = [tokenize_lemmatize(d, nlp, stop_set) for d in cleaned]

    tfidf = TfidfSearcher(tokenized, ids, docs)
    bm25  = BM25Searcher(tokenized, ids, docs)
    return tfidf, bm25, nlp, stop_set

# ---------- UI ----------
st.title("IR Demo: 20 Newsgroups (TF-IDF vs BM25)")
st.caption("Escribe una consulta y usa los botones para ejecutar cada modelo.")

docs, ids = load_docs()
tfidf_searcher, bm25_searcher, nlp, stop_set = build_indices(docs, ids)

query = st.text_input("Consulta en lenguaje natural", value="moon mission")
top_k = st.number_input("Top-K", min_value=1, max_value=10, value=3, step=1)

colA, colB = st.columns(2)
btn_tfidf = colA.button("Buscar con TF-IDF")
btn_bm25  = colB.button("Buscar con BM25")

def render_results(title, results):
    st.subheader(title)
    if not results:
        st.info("Sin resultados.")
        return
    for rank, (doc_id, score, raw) in enumerate(results, start=1):
        st.markdown(f"**#{rank}. {doc_id}** — Score: `{score:.4f}`")
        st.write(raw[:500] + ("..." if len(raw) > 500 else ""))
        st.divider()

if btn_tfidf or btn_bm25:
    q_tokens = tokenize_lemmatize(basic_clean(query), nlp, stop_set)
    if btn_tfidf:
        render_results("Resultados (TF-IDF)", tfidf_searcher.search(q_tokens, top_k=int(top_k)))
    if btn_bm25:
        render_results("Resultados (BM25)", bm25_searcher.search(q_tokens, top_k=int(top_k)))

st.caption("Tip: prueba la misma consulta en ambos botones y compara diferencias.")
