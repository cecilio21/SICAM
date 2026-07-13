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

        col1, col2, col3 = st.columns(3)
        col1.metric("Capítulos CIE-10", df["CAPITULO_CIE10"].nunique())
        col2.metric("Registros", len(df))
        col3.metric("Accuracy", "81.11 %")
        
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
            
        st.info("El sistema SICAM recibe registros de defunción, realiza un proceso de limpieza y normalización textual, transforma el texto mediante TF-IDF, clasifica automáticamente el diagnóstico utilizando LinearSVC y finalmente genera información epidemiológica para apoyar la toma de decisiones.")
        
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
                palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto.lower())
                coincidencias = sum(1 for p in palabras if p in vocabulario_medico)

                if coincidencias == 0:
                    st.error("Diagnóstico no reconocido por el sistema.")
                else:
                    pred = modelo.predict([texto])
                    capitulo_info = cie10[cie10["code"] == pred[0]]
                    
                    if not capitulo_info.empty:
                        descripcion = capitulo_info.iloc[0]["description"]
                        st.info(f"Descripción: {descripcion}")
                    
                    st.session_state.historial.append({
                        "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Diagnóstico": texto,
                        "Resultado": pred[0]
                    })

                    st.markdown(f'''
                    <div style="background-color:white; padding:25px; border-radius:15px; border-left:8px solid #4FC3F7; box-shadow:0px 4px 12px rgba(0,0,0,0.1); margin-top:20px;">
                        <h3 style="color:#2D3958;">🧠 Resultado de la Clasificación</h3>
                        <h2 style="color:#1F2940;">{pred[0]}</h2>
                        <p style="color:green;font-weight:bold;">✔ Clasificación Exitosa</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
        st.markdown("---")
        st.subheader("📋 Historial de Consultas")

        if len(st.session_state.historial) > 0:
            historial_df = pd.DataFrame(st.session_state.historial)
            st.dataframe(historial_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error en la Clasificación Automática: {str(e)}")

def mostrar_explorador_cie10(cie10):
    try:
        st.title("Explorador CIE-10")

        tab1, tab2 = st.tabs([
            "🔎 Buscar por Descripción",
            "🏷️ Buscar por Código"
        ])
            
        with tab1:
            enfermedad = st.text_input("Ingrese enfermedad")
            if st.button("🔍 Buscar", key="buscar_descripcion"):
                resultado = cie10[cie10["description"].astype(str).str.contains(enfermedad, case=False, na=False)]
                if len(resultado) > 0:
                    st.dataframe(resultado[["code","description"]], use_container_width=True)
                else:
                    st.warning("No se encontraron resultados.")
        
        with tab2:
            codigo = st.text_input("Ingrese código CIE-10")
            if st.button("🔍 Buscar", key="buscar_codigo"):
                resultado = cie10[cie10["code"].astype(str).str.contains(codigo, case=False, na=False)]
                if len(resultado) > 0:
                    st.dataframe(resultado[["code","description"]], use_container_width=True)
                else:
                    st.warning("No se encontraron resultados.")
    except Exception as e:
        st.error(f"Error en el Explorador CIE-10: {str(e)}")

def mostrar_validacion_modelo(df):
    try:
        st.title("Validación del Modelo SICAM")
        st.markdown("Evaluación del modelo LinearSVC utilizado para la clasificación automática de causas de muerte según capítulos CIE-10.")

        st.subheader("Métricas Finales")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "81.11%")
        col2.metric("Precision", "82.20%")
        col3.metric("Recall", "81.11%")
        col4.metric("F1 Score", "81.36%")

        st.markdown("---")
        st.subheader("Comparación de Algoritmos")

        metricas = pd.DataFrame({
            "Modelo": ["Naive Bayes", "Logistic Regression", "LinearSVC", "LinearSVC Optimizado"],
            "Accuracy": [0.6616, 0.7555, 0.7729, 0.8111],
            "Precision": [0.6211, 0.7428, 0.7666, 0.8220],
            "Recall": [0.6616, 0.7555, 0.7729, 0.8111],
            "F1": [0.5972, 0.7313, 0.7632, 0.8136]
        })
        st.dataframe(metricas, use_container_width=True)
        
        df_modelos = pd.DataFrame({
            "Modelo": ["Naive Bayes", "Logistic Regression", "LinearSVC", "LinearSVC Optimizado"],
            "Accuracy": [66.16, 75.55, 77.29, 81.11]
        })

        fig = px.bar(df_modelos, x="Modelo", y="Accuracy", text="Accuracy", title="Comparación de Modelos de Machine Learning")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Rendimiento del Modelo")

        df_metricas = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Valor": [81.11, 82.20, 81.11, 81.36]
        })

        fig2 = px.bar(df_metricas, x="Métrica", y="Valor", text="Valor", title="Desempeño Final del Modelo SICAM")
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_layout(yaxis_range=[0,100], height=500)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Matriz de Confusión")
        try:
            st.image("reports/matriz_confusion.png", use_container_width=True)
        except:
            st.warning("No se encontró reports/matriz_confusion.png")

        st.markdown("---")
        st.subheader("Top 10 Diagnósticos Base")
        top10 = df["DIAGBAS_CIEDESC"].value_counts().head(10).reset_index()
        top10.columns = ["Diagnostico", "Cantidad"]

        fig3 = px.bar(top10, x="Cantidad", y="Diagnostico", orientation="h", title="Diagnósticos más frecuentes")
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.subheader("Distribución por Grupos de Edad")
        df["GRUPO_EDAD"] = pd.cut(df["EDAD_ANO"], bins=[0,14,29,44,59,74,120], labels=["0-14", "15-29", "30-44", "45-59", "60-74", "75+"])
        edad = df["GRUPO_EDAD"].value_counts().sort_index().reset_index()
        edad.columns = ["Grupo", "Cantidad"]

        fig4 = px.bar(edad, x="Grupo", y="Cantidad", title="Distribución por Grupo de Edad")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.success("El modelo SICAM obtuvo una precisión global superior al 81%, demostrando una alta capacidad para clasificar automáticamente causas de muerte según los capítulos de la CIE-10.")
    except Exception as e:
        st.error(f"Error en la Validación del Modelo: {str(e)}")

def mostrar_dashboard_epidemiologico(df):
    try:
        st.title("Dashboard Epidemiológico")
        st.subheader("DataSet Defunciones")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Registros", len(df))
        col2.metric("Edad Promedio", round(df["EDAD_ANO"].mean(), 1))
        col3.metric("Capítulos CIE-10", df["CAPITULO_CIE10"].nunique())

        st.subheader("Principales Capítulos CIE-10")
        top10 = df["CAPITULO_CIE10"].value_counts().head(10).reset_index()
        top10.columns = ["Capítulo", "Cantidad"]

        fig = px.bar(top10, x="Cantidad", y="Capítulo", orientation="h", text="Cantidad")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribución por Sexo")
        sexo_df = df["SEXO"].value_counts().reset_index()
        sexo_df.columns = ["Sexo", "Cantidad"]

        fig2 = px.pie(sexo_df, values="Cantidad", names="Sexo")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Distribución por Edad")
        fig3 = px.histogram(df, x="EDAD_ANO", nbins=20)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
    except Exception as e:
        st.error(f"Error en el Dashboard Epidemiológico: {str(e)}")

def mostrar_analitica_avanzada(df):
    try:
        st.title("Analítica Avanzada")

        col1, col2, col3 = st.columns(3)
        col1.metric("Edad Promedio", round(df["EDAD_ANO"].mean(), 1))
        col2.metric("Edad Máxima", int(df["EDAD_ANO"].max()))
        col3.metric("Edad Mínima", int(df["EDAD_ANO"].min()))
              
        st.subheader("Capítulos CIE-10 por Sexo")
        sexo_capitulo = pd.crosstab(df["SEXO"], df["CAPITULO_CIE10"])

        fig = px.imshow(sexo_capitulo, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error en Analítica Avanzada: {str(e)}")

def mostrar_acerca_del_proyecto():
    try:
        st.title("Acerca de SICAM")
        st.write('''
        SICAM es un prototipo desarrollado para la tesis:

        Sistema Automatizado para la Clasificación de Causas de Muerte mediante Aprendizaje Automático en Santo Domingo Norte.

        Autores:
        Cecilio Stalin Rosario Beato y 
        Braulio Disla Torre
        ''')
    except Exception as e:
        st.error(f"Error en Acerca del Proyecto: {str(e)}")