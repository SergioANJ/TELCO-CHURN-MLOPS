"""
simulate_drift.py

Genera un dataset sintético que simula "datos nuevos llegados a producción"
con drift artificial introducido deliberadamente en dos variables:
- Contract: más proporción de "Month-to-month" que en el dataset original
- MonthlyCharges: valores desplazados hacia arriba (simulando subida de precios)

El resto de las variables se mantiene igual a una muestra real del dataset
original, para tener un "grupo de control" que el detector NO debería marcar.
"""

import numpy as np
import pandas as pd

def cargar_datates_original(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)

def simular_drift_contract(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Sobre-representa la categoría 'Month-to-month', simulando una promoción
    reciente de contratos sin permanencia.
    """

    df = df.copy()
    rng = np.random.default_rng(random_state)

    # El 70% de las filas se fuerzan a "Month-to-month",
    # el resto conserva su valor original
    mascara = rng.random(len(df)) < 0.7
    df.loc[mascara, "Contract"] = "Month-to-month"

    return df

def simular_drift_monthly_charges(df: pd.DataFrame, incremento: float = 15.0) -> pd.DataFrame:
    """
    Desplaza MonthlyCharges hacia arriba, simulando una subida general
    de precios del servicio.
    """
    df = df.copy()
    df["MonthlyCharges"] = df["MonthlyCharges"] + incremento

    return df

def generar_datos_con_drift(
        ruta_original: str,
        ruta_salida: str,
        n_muestras: int=500,
        random_state: int=42
) -> pd.DataFrame:
    """
    Orquesta la generación del dataset simulado con drift.
    """
    df_original = cargar_datates_original(ruta_original)

    # Tomamos una muestra (simulando un "lote reciente" de clientes,
    # no el dataset histórico completo)

    df_muestra = df_original.sample(n=n_muestras, random_state=random_state).reset_index(drop=True)

    df_con_drift = simular_drift_contract(df_muestra, random_state=random_state)
    df_con_drift = simular_drift_monthly_charges(df_con_drift, incremento=15.0)

    df_con_drift.to_csv(ruta_salida, index=False)
    print(f"Dataset con drift simulado guardado en: {ruta_salida}")
    print(f"Filas generadas: {len(df_con_drift)}")

    return df_con_drift

if __name__ == "__main__":
    RUTA_ORIGINAL = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    RUTA_SALIDA = "data/monitoring/datos_nuevos_simulados.csv"

    from pathlib import Path
    Path(RUTA_SALIDA).parent.mkdir(parents=True, exist_ok=True)

    generar_datos_con_drift(RUTA_ORIGINAL, RUTA_SALIDA)