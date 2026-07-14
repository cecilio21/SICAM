import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import re

# Importar funciones de la capa modular
from modules import (
    mostrar_inicio,
    mostrar_clasificacion_automatica,
    mostrar_explorador_cie10,
    mostrar_validacion_modelo,
    mostrar_dashboard_epidemiologico,
    mostrar_analitica_avanzada,
    mostrar_acerca_del_proyecto
)

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="SICAM",
    page_icon="🧠",
    layout="wide"
)

# Definir rutas según la arquitectura de tu proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ==========================================
# INYECCIÓN DE ESTILOS CSS (Carga Estricta)
# ==========================================
try:
    # Fuerza la búsqueda estrictamente en la carpeta del archivo actual (app/)
    ruta_css = Path(__file__).resolve().parent / "style.css"
    with open(ruta_css, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"Error al cargar style.css desde la carpeta app: {str(e)}")

# Carga estricta de Datasets
@st.cache_data
def cargar_datos_defunciones():
    return pd.read_csv(DATA_DIR / "defunciones_ml.csv")

@st.cache_data
def cargar_catalogo_cie10():
    return pd.read_csv(DATA_DIR / "cie-10.csv")

# Inicialización de fuentes de datos globales
df = cargar_datos_defunciones()
cie10 = cargar_catalogo_cie10()

# Compilar vocabulario médico extraído del conjunto real
vocabulario_medico = set()
textos_limpios = df["DIAGBAS_CIEDESC"].dropna().astype(str)
for texto in textos_limpios:
    palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto.lower())
    vocabulario_medico.update(palabras)

# Carga estricta de tu modelo LinearSVC binario/multiclase pkl
modelo = joblib.load(MODELS_DIR / "modelo_sicam.pkl")

# Mantener el historial de consultas activo en la sesión
if "historial" not in st.session_state:
    st.session_state.historial = []

# ==========================================
# MENÚ LATERAL (SIDEBAR)
# ==========================================
st.sidebar.markdown('''
<div style="text-align:center; padding:20px 10px;">
    <h1 style="color:white; font-size:48px; font-weight:900; letter-spacing:4px; margin-bottom:10px;">SICAM</h1>
    <p style="color:#B8C7E0; font-size:13px; line-height:1.5;">Sistema Inteligente de Clasificación y Análisis de Mortalidad</p>
    <hr style="border:2px solid #4FC3F7; margin-top:20px;">
</div>
''', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "Inicio",
        "Clasificación Automática",
        "Explorador CIE-10",
        "Validación del Modelo",
        "Dashboard Epidemiológico",
        "Analítica Avanzada",
        "Acerca del Proyecto"
    ]
)

# Orquestador de vistas modulares
if menu == "Inicio":
    mostrar_inicio(df)
elif menu == "Clasificación Automática":
    mostrar_clasificacion_automatica(modelo, cie10, vocabulario_medico)
elif menu == "Explorador CIE-10":
    mostrar_explorador_cie10(cie10)
elif menu == "Validación del Modelo":
    mostrar_validacion_modelo(df)
elif menu == "Dashboard Epidemiológico":
    mostrar_dashboard_epidemiologico(df)
elif menu == "Analítica Avanzada":
    mostrar_analitica_avanzada(df)
elif menu == "Acerca del Proyecto":
    mostrar_acerca_del_proyecto()