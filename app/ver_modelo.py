# app/ver_modelo.py

import joblib

obj = joblib.load("models/modelo_sicam.pkl")

#print(type(obj))
print(obj)

#if isinstance(obj, dict):
    #print("\nCLAVES:")
    #print(obj.keys())
    

#import pandas as pd

#df = pd.read_csv("data/defunciones_ml.csv")

#print(df.columns.tolist())
#print(df["DIAGBAS_CIEDESC"].isna().sum())