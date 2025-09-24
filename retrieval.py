from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

class TfidfSearcher:
    """
    Buscador basado en TF-IDF + similitud coseno.
    Trabaja con documentos ya tokenizados (listas de strings).
    """
    def __init__(self, tokenized_docs: List[List[str]], doc_ids: List[str], raw_docs: List[str]):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,       # identidad (ya tokenizado)
            preprocessor=lambda x: x,    # identidad
            lowercase=False,
            token_pattern=None           # evita warnings cuando pasas tokenizer custom
        )
        self.doc_ids = doc_ids
        self.raw_docs = raw_docs
        self.X = self.vectorizer.fit_transform(tokenized_docs)

    def search(self, query_tokens: List[str], top_k: int = 3) -> List[Tuple[str, float, str]]:
        q = self.vectorizer.transform([query_tokens])
        sims = cosine_similarity(q, self.X).flatten()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i]), self.raw_docs[i]) for i in idxs]

class BM25Searcher:
    """
    Buscador basado en BM25 (Okapi). Premia coincidencia exacta y normaliza por longitud.
    """
    def __init__(self, tokenized_docs: List[List[str]], doc_ids: List[str], raw_docs: List[str]):
        self.bm25 = BM25Okapi(tokenized_docs)  # k1=1.5, b=0.75 por defecto
        self.doc_ids = doc_ids
        self.raw_docs = raw_docs

    def search(self, query_tokens: List[str], top_k: int = 3) -> List[Tuple[str, float, str]]:
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i]), self.raw_docs[i]) for i in idxs]
