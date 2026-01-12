import pandas as pd

def ConvertTypes(df, num_cols, cat_cols, id_col, time_col):
    # 3. CONVERTIR TIPOS EXPLÍCITAMENTE
    # Numéricas -> float
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # errors='coerce' convierte inválidos a NaN
            
    # Categóricas -> string
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Timestamp -> datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # ID -> string
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str)

    return df