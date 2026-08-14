"""
Test Unitarios para src/models/train_model.py
(solo se testean funciones que no reuqieres entrnar un modelo real,
para mantener los test rápidos)

"""

import sys
from pathlib import Path 

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models.train_model import separar_features_target, dividir_train_test

def crear_dataframe_de_prueba() -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": range(20),
        "MonthlyCharges": range(20),
        "Churn": [0,1] * 10 
    })

def test_separar_features_target_no_incluye_target_en_X():
    df = crear_dataframe_de_prueba()
    X, y = separar_features_target(df, "Churn")
    assert "Churn" not in X.columns
    assert y.name == "Churn"


def test_dividir_train_test_respeta_proporcion():
    df = crear_dataframe_de_prueba()
    X, y = separar_features_target(df, "Churn")

    config = {
        "train_test_split": {"test_size": 0.2, "stratify": True},
        "random_state": 42
    }

    X_train, X_test, y_train, y_test = dividir_train_test(X, y, config)

    assert len(X_test) == 4   # 20% de 20 filas
    assert len(X_train) == 16


def test_dividir_train_test_estratifica_correctamente():
    """La proporción de clases en train y test debe ser casi idéntica."""
    df = crear_dataframe_de_prueba()
    X, y = separar_features_target(df, "Churn")

    config = {
        "train_test_split": {"test_size": 0.2, "stratify": True},
        "random_state": 42
    }

    X_train, X_test, y_train, y_test = dividir_train_test(X, y, config)

    proporcion_train = y_train.mean()
    proporcion_test = y_test.mean()

    assert abs(proporcion_train - proporcion_test) < 0.15