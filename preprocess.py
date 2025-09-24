import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
        # Cargar modelo de spaCy para lematización
        try:
            if language == 'spanish':
                self.nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
            else:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            print("Modelo spaCy no encontrado. Usando lematización básica.")
            self.nlp = None
    
    def clean_text(self, text):
        """Limpia y preprocesa el texto"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar signos de puntuación y caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenización
        tokens = word_tokenize(text, language='spanish' if self.language == 'spanish' else 'english')
        
        # Eliminar stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lematización
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            tokens = [token.lemma_ for token in doc]
        
        return tokens
    
    def preprocess_corpus(self, documents):
        """Preprocesa un conjunto de documentos"""
        processed_docs = []
        for doc in documents:
            processed_docs.append(self.clean_text(doc))
        return processed_docs
