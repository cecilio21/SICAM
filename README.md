# SICAM

# Sistema Inteligente de Clasificación y Análisis de Mortalidad

SICAM es un sistema desarrollado como proyecto de tesis de maestría que utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning para clasificar automáticamente las causas de muerte según los capítulos de la Clasificación Internacional de Enfermedades (CIE-10).

---

## Características

- Clasificación automática de causas de muerte.
- Modelo basado en LinearSVC Optimizado.
- Dashboard epidemiológico interactivo.
- Explorador CIE-10.
- Analítica avanzada.
- Visualización de estadísticas de mortalidad.

---

## Tecnologías utilizadas

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- Matplotlib
- Joblib

---

## Modelo de Machine Learning

Modelo seleccionado:

**LinearSVC Optimizado**

Métricas obtenidas:

| Métrica | Valor |
|---------|--------|
| Accuracy | 81.11 % |
| Precision | 82.20 % |
| Recall | 81.11 % |
| F1 Score | 81.36 % |

---

## Dataset

- Registros: 4577
- Variables principales:
  - Diagnóstico básico
  - Diagnósticos asociados
  - Sexo
  - Edad
  - Capítulo CIE-10

---

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app/dashboard.py
```

---

## Autor

Cecilio Rosario
Braulio Disla

Proyecto desarrollado como requisito para optar por el título de Magíster.