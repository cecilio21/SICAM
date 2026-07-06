import pandas as pd

df = pd.read_csv(
    "data/defunciones_ml.csv"
)

print(df.head())

print(df.shape)

print(df["CAPITULO_CIE10"].value_counts())