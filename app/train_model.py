import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ======================================================
# CARGAR DATASET FINAL
# ======================================================
df = pd.read_csv(
    "data/defunciones_ml.csv"
)

# ======================================================
# CONSTRUIR TEXTO COMPLETO
# ======================================================
df["texto_completo"] = (
    df["DIAGBAS_CIEDESC"].fillna("") + " " +
    df["DIAGMUERTE1"].fillna("") + " " +
    df["DIAGMUERTE2"].fillna("") + " " +
    df["DIAGMUERTE3"].fillna("") + " " +
    df["DIAGMUERTE4"].fillna("") + " " +
    df["DIAGMUERTE5"].fillna("") + " " +
    df["DIAGMUERTE6"].fillna("")
)

# ======================================================
# VARIABLES
# ======================================================
X = df["texto_completo"]
y = df["CAPITULO_CIE10"]

# ======================================================
# MODELO DEFINITIVO SICAM
# ======================================================

modelo = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),

    (
        "clf",
        LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    )
])

# ======================================================
# ENTRENAMIENTO
# ======================================================

print("\nEntrenando modelo SICAM...\n")
modelo.fit(X, y)

# ======================================================
# GUARDAR MODELO
# ======================================================

joblib.dump(
    modelo,
    "models/modelo_sicam.pkl"
)

print("==============================================")
print(" MODELO SICAM ENTRENADO CORRECTAMENTE")
print("==============================================")
print("\nModelo guardado en:")
print("models/modelo_sicam.pkl")