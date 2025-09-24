import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

class TfidfSearcher:
    def __init__(self, tokenized_docs, doc_ids, raw_docs):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            lowercase=False
        )
        self.doc_ids = doc_ids
        self.raw_docs = raw_docs
        self.X = self.vectorizer.fit_transform(tokenized_docs)

    def search(self, query_tokens, top_k=3):
        q = self.vectorizer.transform([query_tokens])
        sims = cosine_similarity(q, self.X).flatten()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i]), self.raw_docs[i]) for i in idxs]

class BM25Searcher:
    def __init__(self, tokenized_docs, doc_ids, raw_docs):
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_ids = doc_ids
        self.raw_docs = raw_docs

    def search(self, query_tokens, top_k=3):
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i]), self.raw_docs[i]) for i in idxs]
