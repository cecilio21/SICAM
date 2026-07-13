import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import re

def mostrar_inicio(df):
    try:
        st.markdown('''
        <div class="hero">
            <h1>SICAM</h1>
            <h3>Sistema Inteligente para la Clasificación de Mortalidad</h3>
            <p>Plataforma desarrollada para la clasificación automática de causas de muerte mediante Procesamiento de Lenguaje Natural y Aprendizaje Automático.</p>
        </div>
        ''', unsafe_allow_html=True)

        # Usar las columnas reales de tu dataset para las métricas del Inicio
        capitulos_unicos = df["CAPITULO_CIE10"].nunique() if "CAPITULO_CIE10" in df.columns else 0
        total_registros = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Capítulos CIE-10", capitulos_unicos)
        col2.metric("Registros Analizados", total_registros)
        col3.metric("Accuracy del Modelo", "81.11 %")
        
        st.markdown("---")
        st.subheader("Objetivo del Sistema")
        st.write("Diseñar un sistema automatizado para la clasificación de causas de muerte mediante técnicas de procesamiento de lenguaje natural y aprendizaje automático utilizando Python en Santo Domingo Norte.")
        
        st.markdown("---")
        st.subheader("Funcionalidades Principales")
        st.success("Clasificación automática de diagnósticos médicos.")
        st.success("Explorador interactivo CIE-10.")
        st.success("Dashboard epidemiológico.")
        st.success("Analítica avanzada.")
        
        st.markdown("---")
        st.subheader("Arquitectura del Sistema SICAM")
        st.info("Flujo de procesamiento implementado para la clasificación automática de causas de muerte.")

        c1, c2, c3 = st.columns([2,1,2])
        with c1: st.success("Registro de Defunción")
        with c2: st.markdown("## ➜")
        with c3: st.success("Limpieza y Normalización")

        c1, c2, c3 = st.columns([2,1,2])
        with c1: st.success("Vectorización TF-IDF")
        with c2: st.markdown("## ➜")
        with c3: st.success("Clasificación LinearSVC")

        c1, c2, c3 = st.columns([2,1,2])
        with c1: st.success("Capítulo CIE-10")
        with c2: st.markdown("## ➜")
        with c3: st.success("Dashboard Epidemiológico")
        
        st.markdown("---")
        st.caption("SICAM • Sistema Inteligente para la Clasificación de Mortalidad • Tesis de Maestría • Santo Domingo Norte")
    except Exception as e:
        st.error(f"Error al cargar el módulo Inicio: {str(e)}")

def mostrar_clasificacion_automatica(modelo, cie10, vocabulario_medico):
    try:
        st.title("Clasificación Automática CIE-10")

        diagnostico = st.text_area(
            "Diagnóstico Médico",
            height=150,
            placeholder="Ejemplo: Infarto agudo del miocardio"
        )

        if st.button("Clasificar Diagnóstico"):
            texto = diagnostico.strip()

            if texto == "":
                st.warning("Ingrese un diagnóstico médico.")
            else:
                # Búsqueda flexible en el vocabulario médico compilado de tu dataset
                palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto.lower())
                coincidencias = sum(1 for p in palabras if p in vocabulario_medico)

                if coincidencias == 0 and len(vocabulario_medico) > 0:
                    st.warning("Nota: El diagnóstico contiene palabras no comunes en el historial, procesando de todos modos...")
                
                # Predicción del LinearSVC
                pred = modelo.predict([texto])
                codigo_predicho = pred[0]
                
                # Buscar la descripción en tu dataset cie-10.csv usando tus columnas 'code' y 'description'
                descripcion = "Descripción no encontrada en el catálogo CIE-10."
                if "code" in cie10.columns and "description" in cie10.columns:
                    capitulo_info = cie10[cie10["code"] == codigo_predicho]
                    if not capitulo_info.empty:
                        descripcion = capitulo_info.iloc[0]["description"]
                
                st.info(f"Descripción Asociada: {descripcion}")
                
                st.session_state.historial.append({
                    "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Diagnóstico": texto,
                    "Código Predicho": codigo_predicho
                })

                st.markdown(f'''
                <div style="background-color:white; padding:25px; border-radius:15px; border-left:8px solid #4FC3F7; box-shadow:0px 4px 12px rgba(0,0,0,0.1); margin-top:20px;">
                    <h3 style="color:#2D3958;">🧠 Resultado de la Clasificación</h3>
                    <h2 style="color:#1F2940;">Código CIE-10: {codigo_predicho}</h2>
                    <p style="color:green;font-weight:bold;">✔ Clasificación Exitosa mediante LinearSVC</p>
                </div>
                ''', unsafe_allow_html=True)
                    
        st.markdown("---")
        st.subheader("📋 Historial de Consultas de la Sesión")

        if len(st.session_state.historial) > 0:
            historial_df = pd.DataFrame(st.session_state.historial)
            st.dataframe(historial_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error en la Clasificación Automática: {str(e)}")

def mostrar_explorador_cie10(cie10):
    try:
        st.title("Explorador CIE-10")
        st.markdown("Consulte el diccionario completo de códigos y descripciones médicas cargadas en el sistema.")

        tab1, tab2 = st.tabs([
            "🔎 Buscar por Descripción",
            "🏷️ Buscar por Código"
        ])
            
        with tab1:
            enfermedad = st.text_input("Ingrese término o enfermedad (ej. Infarto, Tumor)")
            if enfermedad:
                resultado = cie10[cie10["description"].astype(str).str.contains(enfermedad, case=False, na=False)]
                if len(resultado) > 0:
                    st.dataframe(resultado[["code", "description"]], use_container_width=True)
                else:
                    st.warning("No se encontraron descripciones que coincidan.")
        
        with tab2:
            codigo = st.text_input("Ingrese código CIE-10 (ej. I21, C00)")
            if codigo:
                resultado = cie10[cie10["code"].astype(str).str.contains(codigo, case=False, na=False)]
                if len(resultado) > 0:
                    st.dataframe(resultado[["code", "description"]], use_container_width=True)
                else:
                    st.warning("No se encontraron códigos que coincidan.")
    except Exception as e:
        st.error(f"Error en el Explorador CIE-10: {str(e)}")

def mostrar_validacion_modelo(df):
    try:
        st.title("Validación del Modelo SICAM")
        st.markdown("Evaluación del modelo **LinearSVC** utilizado para la clasificación automática de causas de muerte según capítulos CIE-10.")

        st.subheader("Métricas Globales de Rendimiento")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "81.11%")
        col2.metric("Precision", "82.20%")
        col3.metric("Recall", "81.11%")
        col4.metric("F1 Score", "81.36%")

        st.markdown("---")
        st.subheader("Comparación de Algoritmos Evaluados")

        metricas = pd.DataFrame({
            "Modelo": ["Naive Bayes", "Logistic Regression", "LinearSVC Base", "LinearSVC Optimizado (SICAM)"],
            "Accuracy": [0.6616, 0.7555, 0.7729, 0.8111],
            "Precision": [0.6211, 0.7428, 0.7666, 0.8220],
            "Recall": [0.6616, 0.7555, 0.7729, 0.8111],
            "F1 Score": [0.5972, 0.7313, 0.7632, 0.8136]
        })
        st.dataframe(metricas, use_container_width=True)
        
        fig = px.bar(metricas, x="Modelo", y="Accuracy", text="Accuracy", title="Comparación de Precisión (Accuracy) entre Modelos")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar gráficos basados en las columnas reales del dataset
        if "DIAGBAS_CIEDESC" in df.columns:
            st.markdown("---")
            st.subheader("Top 10 Diagnósticos Base más Frecuentes en el Dataset")
            top10 = df["DIAGBAS_CIEDESC"].value_counts().head(10).reset_index()
            top10.columns = ["Diagnóstico", "Cantidad"]
            fig_top = px.bar(top10, x="Cantidad", y="Diagnóstico", orientation="h", title="Frecuencia de Diagnósticos")
            st.plotly_chart(fig_top, use_container_width=True)

        if "EDAD_ANO" in df.columns:
            st.markdown("---")
            st.subheader("Distribución por Grupos de Edad")
            df["GRUPO_EDAD"] = pd.cut(df["EDAD_ANO"], bins=[0,14,29,44,59,74,120], labels=["0-14", "15-29", "30-44", "45-59", "60-74", "75+"])
            edad = df["GRUPO_EDAD"].value_counts().sort_index().reset_index()
            edad.columns = ["Grupo de Edad", "Cantidad"]
            fig_edad = px.bar(edad, x="Grupo de Edad", y="Cantidad", title="Casos por Rangos de Edad")
            st.plotly_chart(fig_edad, use_container_width=True)

    except Exception as e:
        st.error(f"Error en la Validación del Modelo: {str(e)}")

def mostrar_dashboard_epidemiologico(df):
    try:
        st.title("Dashboard Epidemiológico")
        st.subheader("Análisis General de Registros de Defunciones")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Registros", len(df))
        if "EDAD_ANO" in df.columns:
            col2.metric("Edad Promedio", round(df["EDAD_ANO"].mean(), 1))
        if "CAPITULO_CIE10" in df.columns:
            col3.metric("Capítulos CIE-10 Detectados", df["CAPITULO_CIE10"].nunique())

        if "CAPITULO_CIE10" in df.columns:
            st.subheader("Distribución de Defunciones por Capítulos CIE-10")
            top_cap = df["CAPITULO_CIE10"].value_counts().head(10).reset_index()
            top_cap.columns = ["Capítulo", "Cantidad"]
            fig_cap = px.bar(top_cap, x="Cantidad", y="Capítulo", orientation="h")
            st.plotly_chart(fig_cap, use_container_width=True)

        if "SEXO" in df.columns:
            st.subheader("Distribución por Sexo")
            sexo_df = df["SEXO"].value_counts().reset_index()
            sexo_df.columns = ["Sexo", "Cantidad"]
            fig_sexo = px.pie(sexo_df, values="Cantidad", names="Sexo", hole=0.4)
            st.plotly_chart(fig_sexo, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error en el Dashboard Epidemiológico: {str(e)}")

def mostrar_analitica_avanzada(df):
    try:
        st.title("Analítica Avanzada")
        
        if "SEXO" in df.columns and "CAPITULO_CIE10" in df.columns:
            st.subheader("Relación Cruzada: Capítulos CIE-10 por Sexo")
            sexo_capitulo = pd.crosstab(df["SEXO"], df["CAPITULO_CIE10"])
            fig = px.imshow(sexo_capitulo, aspect="auto", title="Matriz de Densidad de Diagnósticos")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Las columnas SEXO y CAPITULO_CIE10 son necesarias para desplegar la analítica avanzada.")
    except Exception as e:
        st.error(f"Error en Analítica Avanzada: {str(e)}")

def mostrar_acerca_del_proyecto():
    st.title("Acerca de SICAM")
    st.markdown('''
    ### Sistema Automatizado para la Clasificación de Causas de Muerte mediante Aprendizaje Automático
    Prototipo funcional desarrollado en el marco del proyecto de investigación académica enfocado en el municipio de **Santo Domingo Norte**.

    **Autores del Proyecto:**
    * Cecilio Stalin Rosario Beato
    * Braulio Disla Torre
    
    *Desarrollado con Python, Streamlit, Scikit-Learn y Plotly.*
    ''')