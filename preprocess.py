import re
import unidecode
import spacy
import nltk
from nltk.corpus import stopwords

def _ensure_nltk():
    """Descarga puntualmente los recursos NLTK si faltan (útil en servidores limpios)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

def load_spacy(lang: str = "en"):
    """
    Carga el modelo de spaCy. En producción (Streamlit Cloud), el wheel se instala
    desde requirements.txt; si no está, se descarga on-the-fly.
    """
    if lang != "en":
        raise ValueError("Proyecto configurado para inglés (20 Newsgroups).")
    name = "en_core_web_sm"
    try:
        return spacy.load(name, disable=["parser", "ner", "textcat"])
    except OSError:
        from spacy.cli import download
        download(name)
        return spacy.load(name, disable=["parser", "ner", "textcat"])

def get_stopwords(lang: str = "en"):
    _ensure_nltk()
    if lang != "en":
        raise ValueError("Proyecto configurado para inglés (20 Newsgroups).")
    return set(stopwords.words("english"))

def basic_clean(text: str) -> str:
    """
    Minúsculas, sin acentos, solo letras y espacios, colapsa espacios.
    """
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_lemmatize(text: str, nlp, stop_set):
    """
    Lematiza con spaCy y filtra tokens no alfabéticos y stopwords.
    """
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in stop_set]
    return tokens
