import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def download_20_newsgroups():
    """Descarga y prepara el dataset 20 Newsgroups"""
    from sklearn.datasets import fetch_20newsgroups
    
    print("Descargando 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:20]  # Tomar solo 20 documentos para demo
    return documents

def download_spanish_news():
    """Función alternativa para descargar noticias en español"""
    # Esta es una implementación básica - puedes expandirla
    sample_spanish_docs = [
        "El cambio climático es una amenaza global que requiere acción inmediata de todos los países.",
        "La inteligencia artificial está transformando la manera en que trabajamos y vivimos.",
        "El deporte es fundamental para mantener una vida saludable y equilibrada.",
        "La educación digital se ha vuelto esencial en la era moderna de la tecnología.",
        "La gastronomía española es conocida mundialmente por su diversidad y sabor.",
        "Los avances en medicina permiten tratar enfermedades que antes eran incurables.",
        "El turismo sostenible busca minimizar el impacto ambiental de los viajes.",
        "La literatura latinoamericana tiene representantes de talla mundial como García Márquez.",
        "Las energías renovables son el futuro del abastecimiento energético global.",
        "El emprendimiento tecnológico está en auge en los países de América Latina.",
        "La seguridad cibernética es crucial en un mundo cada vez más conectado.",
        "La música tradicional varía enormemente entre las diferentes regiones de España.",
        "La investigación científica requiere financiamiento constante y apoyo gubernamental.",
        "El transporte público eficiente es clave para reducir la congestión vehicular.",
        "La conservación de especies en peligro de extinción es responsabilidad de todos.",
        "La realidad virtual está revolucionando los sectores del entretenimiento y la educación.",
        "La economía circular promueve la reutilización y reducción de residuos.",
        "Los derechos humanos deben ser protegidos en todas las sociedades democráticas.",
        "La agricultura orgánica gana popularidad entre los consumidores conscientes.",
        "La paz mundial requiere diálogo constante y cooperación internacional."
    ]
    return sample_spanish_docs

def load_custom_documents(file_path):
    """Carga documentos desde un archivo personalizado"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = f.read().split('\n\n')  # Separar documentos por líneas vacías
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        documents = df.iloc[:, 0].tolist()  # Asume que la primera columna contiene el texto
    else:
        raise ValueError("Formato de archivo no soportado")
    
    return documents
