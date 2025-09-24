# preprocess.py  (sin spaCy)
import re
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

def _ensure_nltk():
    """Descarga puntualmente los recursos NLTK si faltan (común en servidores limpios)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

def load_spacy(lang: str = "en"):
    """
    API compatible con tu código original, pero SIN spaCy.
    Devuelve un 'ctx' con herramientas NLTK (lemmatizer y stemmer).
    """
    if lang != "en":
        raise ValueError("Proyecto configurado para inglés (20 Newsgroups).")
    _ensure_nltk()
    return {
        "lemmatizer": WordNetLemmatizer(),
        "stemmer": PorterStemmer()
    }

def get_stopwords(lang: str = "en"):
    _ensure_nltk()
    if lang != "en":
        raise ValueError("Proyecto configurado para inglés (20 Newsgroups).")
    return set(stopwords.words("english"))

def basic_clean(text: str) -> str:
    """
    Minúsculas, sin acentos, deja solo letras y espacios, colapsa espacios.
    """
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_lemmatize(text: str, nlp_ctx, stop_set):
    """
    Tokeniza con NLTK, filtra stopwords y no-alfabéticos,
    y LEMATIZA con WordNet (sin spaCy).
    """
    tokens = []
    for w in word_tokenize(text):
        w = w.lower()
        if w.isalpha() and w not in stop_set:
            w = nlp_ctx["lemmatizer"].lemmatize(w)  # lematización simple
            # Si prefieres stemming: w = nlp_ctx["stemmer"].stem(w)
            tokens.append(w)
    return tokens
