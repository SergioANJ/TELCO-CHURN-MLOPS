"""
Tests unitarios para src/features/bulid_features.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


from features.build_features import (
    codificar_binarias,
    codificar_contract,
    codificar_nominales,
    construir_features,
)

def crear_dataframe_limpio_de_prueba() -> pd.DataFrame:
    """
    Simula la salida de make_dataset.py: ya sin customerID,
    ya sin nulos, pero todavía con categorías en texto.
    """

    return pd.DataFrame({
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "PhoneService": ["Yes", "Yes"],
        "PaperlessBilling": ["Yes", "No"],
        "Churn": ["Yes", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "tenure": [1, 24],
        "MonthlyCharges": [70.35, 56.95],
        "TotalCharges": [70.35, 1366.8],
    })

def test_codificar_binarias_convierte_a_0_1():
    """ Las columnas binarias deben qeudar como enteros: 0/1."""
    df = crear_dataframe_limpio_de_prueba()
    df_cod = codificar_binarias(df)
    assert df_cod["gender"].tolist() == [0, 1]
    assert df_cod["Churn"].tolist() == [1,0]

def test_codificar_binarias_falla_con_valor_inesperado():
    """Si aparece un valor no contemprado (ej. Other), debe fallar."""
    df = crear_dataframe_limpio_de_prueba()
    df.loc[0, "gender"] = "Other"
    with pytest.raises(AssertionError):
        codificar_binarias(df)

def test_codificar_contract_respeta_orden():
    """Month-to-month=0, One year=1, Two year = 2 (Jerarquía real de duracíon)"""
    df = crear_dataframe_limpio_de_prueba()
    df_cod = codificar_contract(df)
    assert df_cod["Contract"].tolist() == [0,2]

def test_codificar_nominales_genera_columnas_dummy():
    """codificar_nominales debe generar columnas nuevas con prefijo del nombre original."""
    df = crear_dataframe_limpio_de_prueba()
    df_cod = codificar_nominales(df)
    columnas_nuevas = [c for c in df_cod.columns if c.startswith("PaymentMethod_")]
    assert len(columnas_nuevas) > 0

def test_construir_features_pipeline_completo_es_numerico():
    """
    Test de integración: el pipeline completo (binarias + contract + nominales)
    debe devolver un DataFrame 100% numérico, sin excepción.
    """
    df = crear_dataframe_limpio_de_prueba()
    df_features = construir_features(df)

    columnas_texto = df_features.select_dtypes(include=["object", "string"]).columns
    assert len(columnas_texto) == 0
    assert df_features.shape[0] == 2  # no se debieron perder filas