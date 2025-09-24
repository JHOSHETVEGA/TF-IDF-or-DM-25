import streamlit as st
import sys
import os

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import TextPreprocessor
from models import VectorialModel, BM25Model
from utils import download_20_newsgroups, download_spanish_news, load_custom_documents

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Recuperación de Información",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Sistema de Recuperación de Información")
    st.markdown("### Implementación de Modelos Vectorial (TF-IDF) y BM25")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Selección de idioma
    language = st.sidebar.selectbox(
        "Idioma del corpus",
        ["english", "spanish"],
        index=1
    )
    
    # Selección de dataset
    dataset_choice = st.sidebar.selectbox(
        "Fuente de documentos",
        ["20 Newsgroups (Inglés)", "Noticias en Español", "Cargar archivo personalizado"]
    )
    
    # Cargar documentos
    documents = []
    
    if dataset_choice == "20 Newsgroups (Inglés)":
        documents = download_20_newsgroups()
    elif dataset_choice == "Noticias en Español":
        documents = download_spanish_news()
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Cargar archivo (TXT o CSV)", 
            type=['txt', 'csv']
        )
        if uploaded_file:
            # Guardar archivo temporalmente
            with open("temp_file", "wb") as f:
                f.write(uploaded_file.getvalue())
            documents = load_custom_documents("temp_file")
            os.remove("temp_file")
    
    if not documents:
        st.warning("Por favor, carga algunos documentos para comenzar.")
        return
    
    # Mostrar información del corpus
    st.sidebar.info(f"📊 Corpus cargado: {len(documents)} documentos")
    
    # Preprocesamiento
    st.header("1. Preprocesamiento de Texto")
    
    with st.spinner("Preprocesando documentos..."):
        preprocessor = TextPreprocessor(language=language)
        processed_docs = preprocessor.preprocess_corpus(documents)
    
    st.success(f"✅ Preprocesamiento completado. {len(processed_docs)} documentos procesados.")
    
    # Mostrar ejemplo de preprocesamiento
    with st.expander("Ver ejemplo de preprocesamiento"):
        st.write("**Documento original:**")
        st.write(documents[0][:200] + "...")
        st.write("**Documento preprocesado:**")
        st.write(" ".join(processed_docs[0][:20]) + "...")
    
    # Entrenar modelos
    st.header("2. Modelos de Recuperación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vectorial_model = VectorialModel()
        vectorial_model.fit(processed_docs, documents)
        st.success("✅ Modelo Vectorial (TF-IDF) entrenado")
    
    with col2:
        bm25_model = BM25Model()
        bm25_model.fit(processed_docs, documents)
        st.success("✅ Modelo BM25 entrenado")
    
    # Búsqueda
    st.header("3. Búsqueda de Documentos")
    
    # Selección de modelo
    model_choice = st.radio(
        "Seleccionar modelo de búsqueda:",
        ["Modelo Vectorial (TF-IDF)", "BM25"]
    )
    
    # Input de consulta
    query = st.text_input(
        "Ingresa tu consulta:",
        placeholder="Ej: inteligencia artificial cambio climático"
    )
    
    if query:
        # Preprocesar consulta
        query_processed = preprocessor.clean_text(query)
        
        if not query_processed:
            st.error("La consulta no contiene términos válidos después del preprocesamiento.")
            return
        
        st.write(f"**Consulta preprocesada:** {', '.join(query_processed)}")
        
        # Realizar búsqueda
        with st.spinner("Buscando documentos relevantes..."):
            if model_choice == "Modelo Vectorial (TF-IDF)":
                results = vectorial_model.search(query_processed, top_k=3)
            else:
                results = bm25_model.search(query_processed, top_k=3)
        
        # Mostrar resultados
        st.header("4. Resultados (Top 3)")
        
        if not results:
            st.warning("No se encontraron documentos relevantes para la consulta.")
        else:
            for i, result in enumerate(results, 1):
                with st.container():
                    st.markdown(f"### 📄 Documento #{i} (Puntaje: {result['score']:.4f})")
                    st.write(f"**Índice en el corpus:** {result['index']}")
                    st.write("**Contenido:**")
                    # Mostrar solo los primeros 300 caracteres
                    content_preview = result['document'][:300] + "..." if len(result['document']) > 300 else result['document']
                    st.write(content_preview)
                    st.markdown("---")
    
    # Información adicional
    st.sidebar.header("Información del Sistema")
    st.sidebar.info("""
    **Características implementadas:**
    - ✅ Preprocesamiento completo
    - ✅ Modelo Vectorial (TF-IDF)
    - ✅ Modelo BM25
    - ✅ Interfaz Streamlit
    - ✅ Soporte para inglés y español
    """)

if __name__ == "__main__":
    # Instalar modelos de spaCy si no están disponibles
    try:
        import spacy
        if len(sys.argv) > 1 and sys.argv[1] == "install-models":
            spacy.cli.download("en_core_web_sm")
            spacy.cli.download("es_core_news_sm")
    except:
        pass
    
    main()
