import re, unidecode
import nltk
from nltk.corpus import stopwords
import spacy

def load_spacy(lang="en"):
    return spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])

def get_stopwords(lang="en"):
    return set(stopwords.words("english"))

def basic_clean(text: str) -> str:
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_lemmatize(text: str, nlp, stop_set):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in stop_set]
