import pandas as pd

# ==========================
# CARGAR DATASET
# ==========================

df = pd.read_csv(
    "data/defunciones_cie10.csv",
    low_memory=False
)

# ==========================
# COLUMNAS A CONSERVAR
# ==========================

columnas = [
    "DIAGBAS_CIEDESC",
    "DIAGMUERTE1",
    "DIAGMUERTE2",
    "DIAGMUERTE3",
    "DIAGMUERTE4",
    "DIAGMUERTE5",
    "DIAGMUERTE6",
    "SEXO",
    "EDAD_ANO",
    "CAPITULO_CIE10"
]

df_ml = df[columnas].copy()

# ==========================
# AGRUPAR CATEGORÍAS RARAS
# ==========================

categorias_raras = [
    "M00-M99 Osteomuscular",
    "O00-O99 Embarazo y parto",
    "H00-H95 Ojo y oído",
    "OTROS",
    "Z00-Z99 Factores de salud",
    "V01-Y98 Causas externas"
]

df_ml["CAPITULO_CIE10"] = df_ml[
    "CAPITULO_CIE10"
].replace(
    categorias_raras,
    "OTRAS_CAUSAS"
)

# ==========================
# LIMPIAR NULOS TEXTO
# ==========================

campos_texto = [
    "DIAGBAS_CIEDESC",
    "DIAGMUERTE1",
    "DIAGMUERTE2",
    "DIAGMUERTE3",
    "DIAGMUERTE4",
    "DIAGMUERTE5",
    "DIAGMUERTE6"
]

for col in campos_texto:
    df_ml[col] = (
        df_ml[col]
        .fillna("")
        .astype(str)
    )

# ==========================
# LIMPIAR SEXO
# ==========================

df_ml["SEXO"] = (
    df_ml["SEXO"]
    .fillna("NO_ESPECIFICADO")
    .astype(str)
)

# ==========================
# LIMPIAR EDAD
# ==========================

df_ml["EDAD_ANO"] = (
    pd.to_numeric(
        df_ml["EDAD_ANO"],
        errors="coerce"
    )
    .fillna(0)
)

# ==========================
# ELIMINAR FILAS SIN CLASE
# ==========================

df_ml = df_ml.dropna(
    subset=["CAPITULO_CIE10"]
)

# ==========================
# GUARDAR DATASET FINAL
# ==========================

df_ml.to_csv(
    "data/defunciones_ml.csv",
    index=False
)

# ==========================
# RESUMEN
# ==========================

print("\nDataset creado correctamente\n")

print("Filas:", len(df_ml))

print("\nDistribución final:\n")

print(
    df_ml["CAPITULO_CIE10"]
    .value_counts()
)