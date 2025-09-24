from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

class VectorialModel:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = None
    
    def fit(self, processed_docs, original_docs):
        """Entrena el modelo TF-IDF"""
        self.documents = original_docs
        # Convertir tokens a texto para TF-IDF
        text_docs = [' '.join(tokens) for tokens in processed_docs]
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(text_docs)
    
    def search(self, query, top_k=3):
        """Realiza búsqueda con el modelo vectorial"""
        if self.vectorizer is None:
            raise ValueError("Modelo no entrenado. Llama al método fit primero.")
        
        # Preprocesar query
        query_vec = self.vectorizer.transform([' '.join(query)])
        
        # Calcular similitud del coseno
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Obtener top-k documentos
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': similarities[idx],
                    'index': idx
                })
        
        return results

class BM25Model:
    def __init__(self):
        self.bm25 = None
        self.documents = None
        self.processed_docs = None
    
    def fit(self, processed_docs, original_docs):
        """Entrena el modelo BM25"""
        self.documents = original_docs
        self.processed_docs = processed_docs
        self.bm25 = BM25Okapi(processed_docs)
    
    def search(self, query, top_k=3):
        """Realiza búsqueda con BM25"""
        if self.bm25 is None:
            raise ValueError("Modelo no entrenado. Llama al método fit primero.")
        
        # Calcular scores BM25
        scores = self.bm25.get_scores(query)
        
        # Obtener top-k documentos
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': scores[idx],
                    'index': idx
                })
        
        return results
