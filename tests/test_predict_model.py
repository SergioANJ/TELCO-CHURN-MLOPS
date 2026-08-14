"""
Test unitarios para src/models/predict_model.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models.predict_model import alinear_columnas, predecir

class ModeloFalso:
    """
    Modelo "de mentira" que simula la interfaz de scikit-learn,
    sin necesidad de entrenar un árbol real. Así el test es rápido
    y no depende de tener un modelo .pkl guardado en disco.
    """
    def predict(self, X):
        # Regla simple y predecible: predice 1 si la primera columna > 50
        return (X.iloc[:, 0] > 50).astype(int).values

    def predict_proba(self, X):
        prob_positiva = (X.iloc[:, 0] > 50).astype(float).values
        prob_negativa = 1 - prob_positiva
        import numpy as np
        return np.column_stack([prob_negativa, prob_positiva])
    
def test_alinear_columnas_columnas_faltantes_con_cero():
    """
    Si al cliente nuevo le falta una columna que el modelo espera
    (ej. ninguno tiene Contract_TwoYear), debe agregarse con valor 0,
    no debe fallar ni desalinear las demás columnas.    
    """
    df_nuevo= pd.DataFrame({"MonthlyCharges": [70.0], "tenure": [5]})
    columnas_entrenamiento = ["MonthlyCharges", "tenure", "Contract_TwoYear"]

    df_alineado = alinear_columnas(df_nuevo, columnas_entrenamiento)

    assert list(df_alineado.columns) == columnas_entrenamiento
    assert df_alineado["Contract_TwoYear"].iloc[0] == 0

def test_alinear_columnas_respeta_el_orden_del_entrenamiento():
    """
    Aunque el Dataframe nuevo tenga las columnas en otro orden, 
    alinear_columnas debe reordenar para que coincidan con el orden exacto
    que el modelo espera (Es crítico para sklearn)
    """
    df_nuevo = pd.DataFrame({"tenure": [5], "MonthlyCharges": [70.0]})
    columnas_entrenamiento = ["MonthlyCharges", "tenure"]

    df_alineado = alinear_columnas(df_nuevo, columnas_entrenamiento)

    assert list(df_alineado.columns) == ["MonthlyCharges", "tenure"]


def test_alinear_columnas_descarta_columnas_extra():
    """
    Si el dato nuevo trae una columna que el modelo NUNCA vio en
    entrenamiento, debe descartarse (no debe llegar al modelo).
    """
    df_nuevo = pd.DataFrame({
        "MonthlyCharges": [70.0],
        "tenure": [5],
        "columna_inventada_que_no_existia": [999]
    })
    columnas_entrenamiento = ["MonthlyCharges", "tenure"]

    df_alineado = alinear_columnas(df_nuevo, columnas_entrenamiento)

    assert "columna_inventada_que_no_existia" not in df_alineado.columns


def test_predecir_devuelve_columnas_esperadas():
    """El DataFrame de salida debe tener prediccion_churn y probabilidad_churn."""
    modelo = ModeloFalso()
    X = pd.DataFrame({"MonthlyCharges": [70.0, 20.0]})

    resultado = predecir(modelo, X)

    assert "prediccion_churn" in resultado.columns
    assert "probabilidad_churn" in resultado.columns
    assert len(resultado) == 2

def test_predecir_probabilidad_entre_0_y_1():
    """Las probabilidades siempre deben estar en el rango válido [0, 1]."""
    modelo = ModeloFalso()
    X = pd.DataFrame({"MonthlyCharges": [70.0, 20.0, 55.0]})

    resultado = predecir(modelo, X)

    assert resultado["probabilidad_churn"].between(0, 1).all()