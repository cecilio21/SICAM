import pandas as pd
import re

# ==========================
# CARGAR DATASET DEFUNCIONES
# ==========================

df = pd.read_csv(
    "data/defunciones.csv",
    low_memory=False
)

# ==========================
# FUNCIÓN CAPÍTULO CIE10
# ==========================

def obtener_capitulo(codigo):

    if pd.isna(codigo):
        return "SIN_CODIGO"

    codigo = str(codigo).upper().strip()

    letra = codigo[0]

    if letra in ["A", "B"]:
        return "A00-B99 Enfermedades infecciosas"

    elif letra == "C":
        return "C00-D49 Neoplasias"

    elif letra == "D":
        return "D50-D89 Enfermedades hematológicas"

    elif letra == "E":
        return "E00-E89 Endocrinas"

    elif letra == "F":
        return "F01-F99 Trastornos mentales"

    elif letra == "G":
        return "G00-G99 Sistema nervioso"

    elif letra == "H":
        return "H00-H95 Ojo y oído"

    elif letra == "I":
        return "I00-I99 Sistema circulatorio"

    elif letra == "J":
        return "J00-J99 Sistema respiratorio"

    elif letra == "K":
        return "K00-K95 Sistema digestivo"

    elif letra == "L":
        return "L00-L99 Piel"

    elif letra == "M":
        return "M00-M99 Osteomuscular"

    elif letra == "N":
        return "N00-N99 Sistema genitourinario"

    elif letra == "O":
        return "O00-O99 Embarazo y parto"

    elif letra == "P":
        return "P00-P96 Periodo perinatal"

    elif letra == "Q":
        return "Q00-Q99 Malformaciones congénitas"

    elif letra == "R":
        return "R00-R99 Hallazgos anormales"

    elif letra in ["S", "T"]:
        return "S00-T98 Traumatismos"

    elif letra in ["V", "W", "X", "Y"]:
        return "V01-Y98 Causas externas"

    elif letra == "Z":
        return "Z00-Z99 Factores de salud"

    else:
        return "OTROS"

# ==========================
# CREAR NUEVA VARIABLE
# ==========================

df["CAPITULO_CIE10"] = df["CIECAUSADEF1"].apply(
    obtener_capitulo
)

# ==========================
# GUARDAR DATASET
# ==========================

df.to_csv(
    "data/defunciones_cie10.csv",
    index=False
)

# ==========================
# MOSTRAR RESULTADOS
# ==========================

print("\nCapítulos encontrados:\n")

print(
    df["CAPITULO_CIE10"]
    .value_counts()
)