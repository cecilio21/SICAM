import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import datetime
import plotly.express as px
import plotly.graph_objects as go
import re

# ==========================
# CONFIGURACIÓN
# ==========================

st.set_page_config(
    page_title="SICAM",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CARGA DE DATOS
# ==========================

#@st.cache_data
#def cargar_datos():
    #return pd.read_csv("data/defunciones_ml.csv")

#df = cargar_datos()

#cie10 = pd.read_csv(
    #"data/cie-10.csv"
#)

# ==========================================
# RUTA BASE DEL PROYECTO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ==========================================
# CARGA DE DATOS
# ==========================================

@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_DIR / "defunciones_ml.csv")

df = cargar_datos()

cie10 = pd.read_csv(DATA_DIR / "cie-10.csv")


#palabras medicas reconocidas
vocabulario_medico = set()

for col in [
    "DIAGBAS_CIEDESC"
]:
    
    textos = df[col].dropna()

    for texto in textos:
        
        palabras = re.findall(
            r"\b[a-záéíóúñ]+\b",
            str(texto).lower()
        )

        vocabulario_medico.update(
            palabras
        )
        

# ==========================
# CARGA DEL MODELO
# ==========================

#modelo = joblib.load(
    #"models/modelo_sicam.pkl"
#)
    modelo = joblib.load(MODELS_DIR / "modelo_sicam.pkl")


if "historial" not in st.session_state:
    st.session_state.historial = []

#estilo css
st.markdown("""
<style>

/* ==========================================
   FONDO PRINCIPAL SICAM
========================================== */

.stApp{

    background: linear-gradient(
        135deg,
        #EEF5FC,
        #DDEBFA,
        #CFE3F8
    );
}

/* ==========================================
   SIDEBAR SICAM
========================================== */

section[data-testid="stSidebar"]{
    background-color:#14243A;
}

/* Título SICAM */
.sidebar-title{
    color:white !important;
}

/* Todo el texto del sidebar */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Opciones del menú */
div[role="radiogroup"] label{

    background-color:transparent !important;

    border-bottom:1px solid rgba(255,255,255,0.15);

    padding-top:12px !important;

    padding-bottom:12px !important;

    margin-bottom:4px !important;

    border-radius:8px;

    transition:all 0.3s ease;
}

/* Texto de cada opción */
div[role="radiogroup"] label p{

    color:white !important;

    font-size:16px !important;

    font-weight:600 !important;

    margin:0 !important;
}

/* Hover */
div[role="radiogroup"] label:hover{

    background-color:#2E4C8C !important;

    cursor:pointer;
}

/* Opción seleccionada */
div[role="radiogroup"] label:has(input:checked){

    background-color:#2F4F7F !important;

    border-left:4px solid #4FC3F7 !important;

    padding-left:12px !important;
}

/* Ocultar círculo del radio */

div[role="radiogroup"] > label > div:first-child{
    display:none !important;
}

/* Ajustar alineación */

div[role="radiogroup"] > label{
    padding-left:15px !important;
}

/* Línea azul bajo el encabezado */
.sidebar-divider{

    border:2px solid #4FC3F7;

    margin-top:20px;
}

/* Título principal */
.sidebar-title{

    color:white !important;

    font-size:52px !important;

    font-weight:900 !important;

    text-align:center;
}

/* Subtítulo */
.sidebar-subtitle{

    color:#D6E4FF !important;

    text-align:center;

    font-size:14px !important;
}


.hero {

    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #2563eb
    );

    padding:40px;

    border-radius:20px;

    text-align:center;

    color:white;

    box-shadow:0px 5px 20px rgba(0,0,0,0.3);

    margin-bottom:20px;
}


/* =====================================
   ARQUITECTURA SICAM
===================================== */

.architecture-container{

    background: linear-gradient(
        135deg,
        #0B1E3A,
        #163B74
    );

    padding:30px;

    border-radius:20px;

    margin-top:20px;

    box-shadow:0px 5px 20px rgba(0,0,0,0.25);
}

.architecture-title{

    text-align:center;

    color:white;

    font-size:32px;

    font-weight:700;

    margin-bottom:10px;
}

.architecture-subtitle{

    text-align:center;

    color:#D6E4FF;

    margin-bottom:35px;
}

.arch-flow{

    display:flex;

    justify-content:center;

    align-items:center;

    flex-wrap:wrap;

    gap:10px;
}

.arch-box{

    background:#1F4E8C;

    color:white;

    padding:20px;

    border-radius:15px;

    min-width:180px;

    text-align:center;

    font-weight:600;

    border:1px solid #4FC3F7;
}

.arch-arrow{

    color:#4FC3F7;

    font-size:35px;

    font-weight:bold;
}

div[data-testid="stMetric"]{

    background:white;

    color:black !important;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

div[data-testid="stMetric"] *{
    color:black !important;
}


div[data-testid="stAlert"]{

    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
text-align:center;
padding:20px 10px;
">

<h1 style="
color:white;
font-size:48px;
font-weight:900;
letter-spacing:4px;
margin-bottom:10px;
">
SICAM
</h1>

<p style="
color:#B8C7E0;
font-size:13px;
line-height:1.5;
">
Sistema Inteligente de Clasificación y
Análisis de Mortalidad
</p>

<hr style="
border:2px solid #4FC3F7;
margin-top:20px;
">

</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "Inicio",
        "Clasificación Automática",
        "Explorador CIE-10",
        "Validación del Modelo",
        "Dashboard Epidemiológico",
        "Analítica Avanzada",
        #"Métricas del Modelo",
        "Acerca del Proyecto"
    ]
)

if menu == "Inicio":        

    st.markdown("""
        <div class="hero">

        <h1>SICAM</h1>

        <h3>
        Sistema Inteligente para la Clasificación de Mortalidad
        </h3>

        <p>
        Plataforma desarrollada para la clasificación automática
        de causas de muerte mediante Procesamiento de Lenguaje
        Natural y Aprendizaje Automático.
        </p>

        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Capítulos CIE-10",
        df["CAPITULO_CIE10"].nunique()
    )

    col2.metric(
        "Registros",
        len(df)
    )

    col3.metric(
        "Accuracy",
        "81.11 %"
    )
    
    ####
    st.markdown("---")

    st.subheader(
        "Objetivo del Sistema"
    )

    st.write(
        """
        Diseñar un sistema automatizado para la clasificación
        de causas de muerte mediante técnicas de procesamiento
        de lenguaje natural y aprendizaje automático utilizando
        Python en Santo Domingo Norte.
        """
    )
    
    #####
    st.markdown("---")

    st.subheader(
        "Funcionalidades Principales"
    )

    st.success(
        "Clasificación automática de diagnósticos médicos."
    )

    st.success(
        "Explorador interactivo CIE-10."
    )

    st.success(
        "Dashboard epidemiológico."
    )

    st.success(
        "Analítica avanzada."
    )
    
    ####
    st.markdown("---")

   
    st.subheader("Arquitectura del Sistema SICAM")

    st.info(
        "Flujo de procesamiento implementado para la clasificación automática de causas de muerte."
    )

    c1, c2, c3 = st.columns([2,1,2])

    with c1:
        st.success("Registro de Defunción")

    with c2:
        st.markdown("## ➜")

    with c3:
        st.success("Limpieza y Normalización")

    c1, c2, c3 = st.columns([2,1,2])

    with c1:
        st.success("Vectorización TF-IDF")

    with c2:
        st.markdown("## ➜")

    with c3:
        st.success("Clasificación LinearSVC")

    c1, c2, c3 = st.columns([2,1,2])

    with c1:
        st.success("Capítulo CIE-10")

    with c2:
        st.markdown("## ➜")

    with c3:
        st.success("Dashboard Epidemiológico")
        
    st.info("""
        El sistema SICAM recibe registros de defunción, realiza un proceso de limpieza
        y normalización textual, transforma el texto mediante TF-IDF, clasifica
        automáticamente el diagnóstico utilizando LinearSVC y finalmente genera
        información epidemiológica para apoyar la toma de decisiones.
        """)
    
    ####
    st.markdown("---")

    st.caption(
        "SICAM • Sistema Inteligente para la Clasificación de Mortalidad • Tesis de Maestría • Santo Domingo Norte"
    )
    
if menu == "Clasificación Automática":

    st.title(
        "Clasificación Automática CIE-10"
    )

    diagnostico = st.text_area(
        "Diagnóstico Médico",
        height=150,
        placeholder="Ejemplo: Infarto agudo del miocardio"
    )

 ###
    palabras = re.findall(
        r"\b[a-záéíóúñ]+\b",
        texto.lower()
    )

    coincidencias = sum(
        1 for p in palabras
        if p in vocabulario_medico
    )
    ###

    if st.button("Clasificar Diagnóstico"):

        texto = diagnostico.strip()

        if texto == "":
            st.warning(
                "Ingrese un diagnóstico médico."
            )

        else:

            palabras = re.findall(
                r"\b[a-záéíóúñ]+\b",
                texto.lower()
            )

            coincidencias = sum(
                1 for p in palabras
                if p in vocabulario_medico
            )

            #st.write("Palabras encontradas:", palabras)
            #st.write("Coincidencias:", coincidencias)

            if coincidencias == 0:

                st.error(
                    "Diagnóstico no reconocido por el sistema."
                )

            else:

                pred = modelo.predict([texto])
                
                #####
                capitulo_info = cie10[
                    cie10["code"] == pred[0]]
                
                if not capitulo_info.empty:

                    descripcion = (
                        capitulo_info
                        .iloc[0]["description"])

                    st.info(
                        f"Descripción: {descripcion}")
                #####
                
                st.session_state.historial.append({
                    "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Diagnóstico": texto,
                    "Resultado": pred[0]
                })
                
                scores = modelo.decision_function([texto])

                #confianza = round(
                #    scores.max() * 10,
                #    2
                #)
                #<p style="font-size:18px;color:#4F7CFF;">
                    #Nivel de confianza: {confianza}%
                #</p>

                st.markdown(f"""
                <div style="
                background-color:white;
                padding:25px;
                border-radius:15px;
                border-left:8px solid #4FC3F7;
                box-shadow:0px 4px 12px rgba(0,0,0,0.1);
                margin-top:20px;
                ">

                <h3 style="color:#2D3958;">
                🧠 Resultado de la Clasificación
                </h3>

                

                <h2 style="color:#1F2940;">
                {pred[0]}
                </h2>

                <p style="color:green;font-weight:bold;">
                ✔ Clasificación Exitosa
                </p>

                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")

        st.subheader(
            "📋 Historial de Consultas"
        )

        if len(st.session_state.historial) > 0:

            historial_df = pd.DataFrame(
                st.session_state.historial
            )

            st.dataframe(
                historial_df,
                use_container_width=True
            )

if menu == "Explorador CIE-10":

    st.title(
        "Explorador CIE-10"
    )

    tab1, tab2 = st.tabs([
    "🔎 Buscar por Descripción",
    "🏷️ Buscar por Código"
    ])
        
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
    background-color: #1F2940;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: white !important;
        font-weight: 600 !important;
    }
    <style>
    """, unsafe_allow_html=True)
            
        #busqueda por descripcion
    with tab1:

        enfermedad = st.text_input(
        "Ingrese enfermedad"
        )

        if st.button(
            "🔍 Buscar",
            key="buscar_descripcion"
        ):

            resultado = cie10[
                cie10["description"]
                .astype(str)
                .str.contains(
                    enfermedad,
                    case=False,
                    na=False
                )
            ]

            if len(resultado) > 0:

                st.dataframe(
                    resultado[
                        ["code","description"]
                    ],
                    use_container_width=True
                )

            else:

                st.warning(
                    "No se encontraron resultados."
                )
    
     #busqueda por codigo
    with tab2:

        codigo = st.text_input(
            "Ingrese código CIE-10"
        )

        if st.button(
            "🔍 Buscar",
            key="buscar_codigo"
        ):

            resultado = cie10[
                cie10["code"]
                .astype(str)
                .str.contains(
                    codigo,
                    case=False,
                    na=False
                )
            ]

            if len(resultado) > 0:

                st.dataframe(
                    resultado[
                        ["code","description"]
                    ],
                    use_container_width=True
                )

            else:

                st.warning(
                    "No se encontraron resultados."
                )
      
if menu == "Validación del Modelo":
   
    # =====================================================
    # VALIDACIÓN DEL MODELO
    # =====================================================

    st.title("Validación del Modelo SICAM")

    st.markdown("""
    Evaluación del modelo LinearSVC utilizado para la clasificación
    automática de causas de muerte según capítulos CIE-10.
    """)

    # =====================================
    # MÉTRICAS
    # =====================================

    st.subheader("Métricas Finales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "81.11%"
    )

    col2.metric(
        "Precision",
        "82.20%"
    )

    col3.metric(
        "Recall",
        "81.11%"
    )

    col4.metric(
        "F1 Score",
        "81.36%"
    )

    st.markdown("---")

    # =====================================
    # COMPARACIÓN DE MODELOS
    # =====================================

    st.subheader("Comparación de Algoritmos")

    #Tabla de comparacion
    metricas = pd.DataFrame({
    
        "Modelo":[
            "Naive Bayes",
            "Logistic Regression",
            "LinearSVC",
            "LinearSVC Optimizado"
        ],
    
        "Accuracy":[
            0.6616,
            0.7555,
            0.7729,
             0.8111
        ],
    
        "Precision":[
            0.6211,
            0.7428,
            0.7666,
            0.8220
        ],
    
        "Recall":[
            0.6616,
            0.7555,
            0.7729,
            0.8111
        ],
    
        "F1":[
            0.5972,
            0.7313,
            0.7632,
            0.8136
        ]
    })
    
    st.dataframe(
        metricas,
        use_container_width=True
    )
    
    df_modelos = pd.DataFrame({
        "Modelo":[
            "Naive Bayes",
            "Logistic Regression",
            "LinearSVC",
            "LinearSVC Optimizado"
        ],
        
        "Accuracy":[
            66.16,
            75.55,
            77.29,
            81.11
        ]
    })

    fig = px.bar(
        df_modelos,
        x="Modelo",
        y="Accuracy",
        text="Accuracy",
        title="Comparación de Modelos de Machine Learning"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # MÉTRICAS DEL MODELO
    # =====================================

    st.subheader("Rendimiento del Modelo")

    df_metricas = pd.DataFrame({
        "Métrica":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Valor":[
            81.11,
            82.20,
            81.11,
            81.36
        ]
    })

    fig = px.bar(
        df_metricas,
        x="Métrica",
        y="Valor",
        text="Valor",
        title="Desempeño Final del Modelo SICAM"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0,100],
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # MATRIZ DE CONFUSIÓN
    # =====================================

    st.subheader("Matriz de Confusión")

    try:

        st.image(
            "reports/matriz_confusion.png",
            use_container_width=True
        )

    except:

        st.warning(
            "No se encontró reports/matriz_confusion.png"
        )

    st.markdown("---")

    # =====================================
    # TOP 10 DIAGNÓSTICOS
    # =====================================

    st.subheader("Top 10 Diagnósticos Base")

    top10 = (
        df["DIAGBAS_CIEDESC"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top10.columns = [
        "Diagnostico",
        "Cantidad"
    ]

    fig = px.bar(
        top10,
        x="Cantidad",
        y="Diagnostico",
        orientation="h",
        title="Diagnósticos más frecuentes"
    )

    fig.update_layout(
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # DISTRIBUCIÓN POR EDAD
    # =====================================

    st.subheader("Distribución por Grupos de Edad")

    df["GRUPO_EDAD"] = pd.cut(
        df["EDAD_ANO"],
        bins=[0,14,29,44,59,74,120],
        labels=[
            "0-14",
            "15-29",
            "30-44",
            "45-59",
            "60-74",
            "75+"
        ]
    )

    edad = (
        df["GRUPO_EDAD"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    edad.columns = [
        "Grupo",
        "Cantidad"
    ]

    fig = px.bar(
        edad,
        x="Grupo",
        y="Cantidad",
        title="Distribución por Grupo de Edad"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.success("""
    El modelo SICAM obtuvo una precisión global superior al 81%,
    demostrando una alta capacidad para clasificar automáticamente
    causas de muerte según los capítulos de la CIE-10.
    """)

if menu == "Dashboard Epidemiológico":
    
    #Grafico 1
    st.title(
    "Dashboard Epidemiológico"
    )
    
    st.subheader(
        "DataSet Defunciones"
    )
    
    col1,col2,col3 = st.columns(3)
    
    col1.metric(
        "Registros",
        len(df)
    )
    
    col2.metric(
        "Edad Promedio",
         round(df["EDAD_ANO"].mean(),1)
    )
    
    col3.metric(
        "Capítulos CIE-10",
        df["CAPITULO_CIE10"].nunique()
    )

    #Grafico 3
    st.subheader(
    "Principales Capítulos CIE-10"
    )

    top10 = (
        df["CAPITULO_CIE10"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top10.columns = [
        "Capítulo",
        "Cantidad"
    ]

    fig = px.bar(
        top10,
        x="Cantidad",
        y="Capítulo",
        orientation="h",
        text="Cantidad"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #grafico 2
    st.subheader(
        "Distribución por Sexo"
    )

    sexo_df = (
        df["SEXO"]
        .value_counts()
        .reset_index()
    )

    sexo_df.columns = [
        "Sexo",
        "Cantidad"
    ]

    fig = px.pie(
        sexo_df,
        values="Cantidad",
        names="Sexo"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    

    #Grafico 3
    st.subheader(
        "Distribución por Edad"
    )

    fig = px.histogram(
        df,
        x="EDAD_ANO",
        nbins=20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")
       
if menu == "Analítica Avanzada":

    st.title(
        "Analítica Avanzada"
    )

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Edad Promedio",
        round(df["EDAD_ANO"].mean(),1)
    )

    col2.metric(
        "Edad Máxima",
        int(df["EDAD_ANO"].max())
    )

    col3.metric(
        "Edad Mínima",
        int(df["EDAD_ANO"].min())
    )
          
    #antc 3
    st.subheader(
    "Capítulos CIE-10 por Sexo"
    )

    sexo_capitulo = pd.crosstab(
        df["SEXO"],
        df["CAPITULO_CIE10"]
    )

    fig = px.imshow(
        sexo_capitulo,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

#pendiente de quitar
if menu == "Métricas del Modelo":

    st.title(
        "Evaluación de Modelos"
    )

    metricas = pd.DataFrame({

        "Modelo":[
            "Naive Bayes",
            "Logistic Regression",
            "LinearSVC"
        ],

        "Accuracy":[
            0.6616,
            0.7555,
            0.7729
        ],

        "Precision":[
            0.6211,
            0.7428,
            0.7666
        ],

        "Recall":[
            0.6616,
            0.7555,
            0.7729
        ],

        "F1":[
            0.5972,
            0.7313,
            0.7632
        ]
    })

    st.dataframe(
        metricas,
        use_container_width=True
    )

    fig = px.bar(
        metricas,
        x="Modelo",
        y="Accuracy"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
       
if menu == "Acerca del Proyecto":

    st.title(
        "Acerca de SICAM"
    )

    st.write("""
    SICAM es un prototipo desarrollado para la tesis:

    Sistema Automatizado para la Clasificación
    de Causas de Muerte mediante Aprendizaje
    Automático en Santo Domingo Norte.

    Autores:
    Cecilio Stalin Rosario Beato y 
    Braulio Disla Torre
    """)
    
    