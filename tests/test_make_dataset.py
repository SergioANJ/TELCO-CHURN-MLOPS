"""
Test unitarios para src/data/make_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import limpiar_datos

def crear_dataframe_de_prueba() -> pd.DataFrame:
    """
    Crea un mini-dataset sintético que simula el problema real:
    un cliente con ternure=0 y TotalCharges vacío (" " ), y otro normal.
    """
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB"],
        "gender": ["Female", "Male"],
        "tenure": [0, 12],
        "TotalCharges": [" ", "500.5"],
        "MonthlyCharges": [29.85, 56.95],
    })

def test_limpiar_datos_elimina_customer_id():
    """customerID no debe existir en el DataFrame de salida."""
    df = crear_dataframe_de_prueba()
    df_limpio = limpiar_datos(df)
    assert "customerID" not in df_limpio.columns

def test_limpiar_datos_convierte_total_charges_a_numerico():
    """TotalCharges debe quedar como tipo numérico (Float), no texto"""
    df = crear_dataframe_de_prueba()
    df_limpio = limpiar_datos(df)
    assert pd.api.types.is_numeric_dtype(df_limpio["TotalCharges"])

def test_limpiar_datos_imputa_nulos_con_cero():
    """El cliente con ternure=0 y TotalCharges=' ' debe quedar con TotalCharges=0."""
    df = crear_dataframe_de_prueba()
    df_limpio = limpiar_datos(df)
    assert df_limpio.loc[0,"TotalCharges"] == 0.0

def test_limpiar_datos_no_deja_nulos():
    """Despues de limpiar, no debe quedar ningún nulo en el Dataframe"""
    df=crear_dataframe_de_prueba()
    df_limpio = limpiar_datos(df)
    assert df_limpio.isnull().sum().sum() == 0

def test_limpiar_datos_falla_si_premisa_de_negocio_no_se_cumple():
    """
    Si hay un nulo en TotalCharges que NO corresponde a ternure =0,
    la funcion debe fallar (assert), no imputar silenciosamente.
    """
    df = pd.DataFrame({
        "customerID": ["0003-CCC"],
        "gender": ["Female"],
        "tenure": [24],          # tenure != 0
        "TotalCharges": [" "],   # pero TotalCharges viene vacío
        "MonthlyCharges": [70.0], 
    })

    with pytest.raises(AssertionError):
        limpiar_datos(df)