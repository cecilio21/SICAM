import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import re

# Importar funciones modulares
from modules import (
    mostrar_inicio,
    mostrar_clasificacion_automatica,
    mostrar_explorador_cie10,
    mostrar_validacion_modelo,
    mostrar_dashboard_epidemiologico,
    mostrar_analitica_avanzada,
    mostrar_acerca_del_proyecto
)

# Configuración de página
st.set_page_config(
    page_title="SICAM",
    page_icon="🧠",
    layout="wide"
)

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Cargar estilos CSS externos
try:
    with open(BASE_DIR / "style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning("No se pudo cargar el archivo style.css")

# Carga de datos con manejo de excepciones y placeholders si no existen archivos locales
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv(DATA_DIR / "defunciones_ml.csv")
    except Exception:
        # Generar datos simulados para asegurar funcionamiento local
        return pd.DataFrame({
            "CAPITULO_CIE10": ["Capítulo I", "Capítulo II", "Capítulo IX", "Capítulo I"],
            "DIAGBAS_CIEDESC": ["Infarto agudo", "Neoplasia", "Hipertensión", "Infección"],
            "EDAD_ANO": [65, 42, 78, 55],
            "SEXO": ["M", "F", "M", "F"]
        })

@st.cache_data
def cargar_cie10():
    try:
        return pd.read_csv(DATA_DIR / "cie-10.csv")
    except Exception:
        return pd.DataFrame({
            "code": ["I21", "C00", "I10"],
            "description": ["Infarto agudo del miocardio", "Tumor maligno", "Hipertensión esencial"]
        })

df = cargar_datos()
cie10 = cargar_cie10()

# Inicializar vocabulario médico
vocabulario_medico = set()
for col in ["DIAGBAS_CIEDESC"]:
    if col in df.columns:
        textos = df[col].dropna()
        for texto in textos:
            palabras = re.findall(r"\b[a-záéíóúñ]+\b", str(texto).lower())
            vocabulario_medico.update(palabras)

# Carga del modelo
class ModeloDummy:
    def predict(self, X): return ["I21"]
    def decision_function(self, X): return [[1.0]]

try:
    modelo = joblib.load(MODELS_DIR / "modelo_sicam.pkl")
except Exception:
    modelo = ModeloDummy()

if "historial" not in st.session_state:
    st.session_state.historial = []

# Sidebar - Cabecera
st.sidebar.markdown('''
<div style="text-align:center; padding:20px 10px;">
    <h1 style="color:white; font-size:48px; font-weight:900; letter-spacing:4px; margin-bottom:10px;">SICAM</h1>
    <p style="color:#B8C7E0; font-size:13px; line-height:1.5;">Sistema Inteligente de Clasificación y Análisis de Mortalidad</p>
    <hr style="border:2px solid #4FC3F7; margin-top:20px;">
</div>
''', unsafe_allow_html=True)

# Menú de navegación
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

# Llamadas estructuradas a funciones
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