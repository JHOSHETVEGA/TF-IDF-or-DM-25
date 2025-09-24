# preprocess.py
import re, unidecode, spacy, nltk
from nltk.corpus import stopwords

def _ensure_nltk():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')

def load_spacy(lang: str = "en"):
    if lang != "en":
        raise ValueError("Proyecto configurado para inglés (20 Newsgroups).")
    name = "en_core_web_sm"
    try:
        return spacy.load(name, disable=["parser","ner","textcat"])
    except OSError:
        # Descarga automática si no está instalado en el servidor
        from spacy.cli import download
        download(name)
        return spacy.load(name, disable=["parser","ner","textcat"])

def get_stopwords(lang: str = "en"):
    _ensure_nltk()
    return set(stopwords.words("english"))

def basic_clean(text: str) -> str:
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_lemmatize(text: str, nlp, stop_set):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in stop_set]
