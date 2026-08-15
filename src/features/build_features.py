"""
build_feature.py

convierte el dataset limpio (processd) en un dataset totalment numérico,
listo para entrenar el modelo.

Estrategia de encoding documentada (ver notebooks/01_eda.ipynb Paso 3.5)

1. Variables binarias: mapeo manual 0/1
2. Contract: Ordinal Encoding (existe jerarquia real de duración)
3. Resto de categóricas nominales: One-Hot Encoding con drop_first=True
"""

import pandas as pd


# Constantes de mapeo — centralizadas aquí para que sean fáciles de
# encontrar y modificar, en vez de estar "escondidas" dentro de la función
MAPEOS_BINARIOS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'Churn': {'No': 0, 'Yes': 1}
}

MAPEO_CONTRACT = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

COLUMNAS_ONE_HOT = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod'
]

def codificar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica mapeo manual 0/1 a las variables binarias definidas en MAPEOS_BINARIOS."""
    df = df.copy()
    for col, mapeo in MAPEOS_BINARIOS.items():
        if col not in df.columns:
             continue  # normal en inferencia: Churn no existe todavía
        
        valores_esperados = set(mapeo.keys())
        valores_reales = set(df[col].unique())
        assert valores_reales.issubset(valores_esperados), (
            f"La columna '{col}' tiene valores inesperados: "
            f"{valores_reales - valores_esperados}"
        )
        df[col] = df[col].map(mapeo)
    return df


def codificar_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica Ordinal Encoding a Contract, respetando el orden real de duración."""
    df = df.copy()
    valores_esperados = set(MAPEO_CONTRACT.keys())
    valores_reales = set(df['Contract'].unique())
    assert valores_reales.issubset(valores_esperados), (
        f"Contract tiene valores inesperados: {valores_reales - valores_esperados}"
    )
    df['Contract'] = df['Contract'].map(MAPEO_CONTRACT)
    return df

def codificar_nominales(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding (drop_first=True) a las categóricas sin orden natural."""
    df = pd.get_dummies(df, columns=COLUMNAS_ONE_HOT, drop_first=True)

    # get_dummies puede generar columnas bool; forzamos a int por consistencia
    cols_bool = df.select_dtypes(include='bool').columns
    df[cols_bool] = df[cols_bool].astype(int)

    return df

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquesta el pipeline completo de encoding.

    Args:
        df: DataFrame limpio (salida de make_dataset.py).

    Returns:
        DataFrame 100% numérico, listo para separar en X/y y entrenar.
    """
    df = codificar_binarias(df)
    df = codificar_contract(df)
    df = codificar_nominales(df)

    # Verificación final: no debe quedar ninguna columna de texto
    columnas_texto = df.select_dtypes(include=['object', 'string']).columns.tolist()
    assert len(columnas_texto) == 0, (
        f"Quedaron columnas sin codificar: {columnas_texto}"
    )

    return df

def guardar_features(df: pd.DataFrame, ruta_salida: str) -> None:
    """Guarda el dataset con features listas para modelar."""
    df.to_csv(ruta_salida, index=False)
    print(f"Features guardadas en: {ruta_salida}")

def main(ruta_processed: str, ruta_features: str) -> pd.DataFrame:
    """Orquesta la carga del dataset limpio y la construcción de features."""
    df_limpio = pd.read_csv(ruta_processed)
    df_features = construir_features(df_limpio)
    guardar_features(df_features, ruta_features)
    return df_features

if __name__ == "__main__":
    RUTA_PROCESSED = "data/processed/telco_churn_clean.csv"
    RUTA_FEATURES = "data/processed/telco_churn_features.csv"

    main(RUTA_PROCESSED, RUTA_FEATURES)