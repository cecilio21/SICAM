import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ======================================================
# CARGAR DATASET
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
# TRAIN / TEST
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================================
# MODELOS A COMPARAR
# ======================================================

modelos = {

    "Naive Bayes":
        Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ]),

    "Logistic Regression":
        Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

    "LinearSVC":
        Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf",
                LinearSVC(
                    random_state=42
                )
            )
        ]),

    "LinearSVC Optimizado":
        Pipeline([
            ("tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1,2),
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            ("clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])
}
# ======================================================
# ENTRENAMIENTO
# ======================================================

resultados = []
for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    inicio = time.time()

    modelo.fit(
        X_train,
        y_train
    )

    tiempo = time.time() - inicio

    y_pred = modelo.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    resultados.append([
        nombre,
        round(accuracy,4),
        round(precision,4),
        round(recall,4),
        round(f1,4),
        round(tiempo,2)
    ])

# ======================================================
# TABLA FINAL
# ======================================================

resultado_df = pd.DataFrame(
    resultados,

    columns=[
        "Modelo",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Tiempo (s)"
    ]
)

resultado_df = resultado_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n")
print("="*90)
print("COMPARACIÓN DE MODELOS")
print("="*90)
print(resultado_df)

# ======================================================
# GUARDAR RESULTADOS
# ======================================================

resultado_df.to_csv(
    "reports/comparacion_modelos.csv",
    index=False
)

print(
    "\nArchivo guardado en reports/comparacion_modelos.csv"
)