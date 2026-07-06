import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# MODELO DEFINITIVO SICAM
# ======================================================

modelo = Pipeline([

    (

        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            ngram_range=(1,2),

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

print("\nEntrenando modelo...")

modelo.fit(

    X_train,

    y_train

)

# ======================================================
# PREDICCIÓN
# ======================================================

y_pred = modelo.predict(

    X_test

)

# ======================================================
# MÉTRICAS
# ======================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n==============================")
print("VALIDACIÓN DEL MODELO SICAM")
print("==============================")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ======================================================
# REPORTE DE CLASIFICACIÓN
# ======================================================

print("\n==============================")
print("REPORTE DE CLASIFICACIÓN")
print("==============================\n")

print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)

# ======================================================
# MATRIZ DE CONFUSIÓN
# ======================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

fig, ax = plt.subplots(

    figsize=(15,13)

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=modelo.classes_

)

disp.plot(

    cmap="Blues",

    xticks_rotation=90,

    values_format="d",

    colorbar=False,

    ax=ax

)

plt.title(

    "Matriz de Confusión - LinearSVC Optimizado",

    fontsize=16,

    fontweight="bold"

)

plt.xlabel("Clase Predicha")

plt.ylabel("Clase Real")

plt.tight_layout()

plt.savefig(

    "reports/matriz_confusion_linearsvc_optimizado.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print(

    "\nMatriz guardada en reports/matriz_confusion_linearsvc_optimizado.png"

)