import pandas as pd
import numpy as np
from Functions import ConvertTypes
#low memory porque hay distintos tipos de datos en varias columnas
df = pd.read_csv(r"C:\Users\pedro\Transformers4Rec\202512_data_train_SC_GMV.csv", low_memory=False)

#Renombro para que sea consistente el nombre de las columnas
df.rename(columns={'    SIT_SITE_ID': 'SIT_SITE_ID'}, inplace=True)
# 1. Defino columnas
id_col = 'CUS_CUST_ID_SEL'
time_col = 'SELLER_TIMESTAMP'
action_col = 'CARD_KEY_CLICK'
reward_col = 'REWARD'

state_features = [
    'SIT_SITE_ID', 'SELLER_SEGMENT', 'REP_CURRENT_LEVEL', # Categóricas
    'ORDERS', 'GMV_USD', 'VISITS_30D', 'CVR_30D'          # Numéricas
]

# Filtrar solo lo que sirve (para liberar memoria RAM)
subset_cols = [id_col, time_col, action_col, reward_col] + state_features
print(subset_cols)
df_clean = df[subset_cols].copy()

# Ordenar CRUCIAL: Por usuario y por tiempo
df_clean = df_clean.sort_values(by=[id_col, time_col])

# Estandarizar tipos de datos en todas las columnas
num_cols = ['ORDERS', 'GMV_USD', 'VISITS_30D', 'CVR_30D']
cat_cols = ['SIT_SITE_ID', 'SELLER_SEGMENT', 'REP_CURRENT_LEVEL']

df = ConvertTypes.ConvertTypes(df, num_cols, cat_cols, id_col, time_col)

# Manejo básico de nulos (IMPORTANTE: El Transformer odia los NaNs)
# Aquí asumo ceros para numéricos y 'UNKNOWN' para textos, pero valídalo.
df_clean[num_cols] = df_clean[num_cols].fillna(0)
df_clean[cat_cols] = df_clean[cat_cols].fillna('UNKNOWN')

print("Datos ordenados y limpios. Filas:", df_clean.shape[0])