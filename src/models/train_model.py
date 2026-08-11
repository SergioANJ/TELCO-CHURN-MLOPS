"""
train_model.py

Entrena el modelo campeón (Árbol de Decisión con class_weight='balanced')
usando los hiperparámetros óptimos encontrados en la fase de experimentación
(ver notebooks/01_eda.ipynb y los 4 runs comparados en MLflow).

Este script reproduce el modelo ganador de forma automatizada y trackeada.
"""

import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)

def cargar_config(ruta_config: str = "config.yaml") -> dict:
    """Carga la configuración centralizada del proyecto."""
    with open(ruta_config, "r") as f:
        config = yaml.safe_load(f)
    return config

def separar_features_target(df: pd.DataFrame, target_column: str):
    """Separa el DataFrame en X (features) e y (target)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def dividir_train_test(X, y, config: dict):
    """Realiza el train/test split respetando la configuración (estratificado)."""
    stratify_param = y if config["train_test_split"]["stratify"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train_test_split"]["test_size"],
        random_state=config["random_state"],
        stratify=stratify_param
    )
    return X_train, X_test, y_train, y_test

def entrenar_con_gridsearch(X_train, y_train, config: dict) -> GridSearchCV:
    """Ejecuta la búsqueda de hiperparámetros con GridSearchCV, usando class_weight fijo."""
    modelo_base = DecisionTreeClassifier(
        random_state=config["random_state"],
        class_weight=config["model"]["class_weight"]
    )

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=config["grid_search"]["param_grid"],
        scoring=config["grid_search"]["scoring"],
        cv=config["grid_search"]["cv"],
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search

def evaluar_modelo(modelo, X_test, y_test) -> dict:
    """Calcula el set completo de métricas sobre el conjunto de prueba."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy_test": accuracy_score(y_test, y_pred),
        "recall_test": recall_score(y_test, y_pred),
        "precision_test": precision_score(y_test, y_pred),
        "f1_test": f1_score(y_test, y_pred),
        "roc_auc_test": roc_auc_score(y_test, y_proba),
    }
    return metricas

def guardar_modelo_local(modelo, ruta_salida: str) -> None:
    """Guarda una copia local del modelo (independiente de MLflow)."""
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_salida)
    print(f"Modelo guardado localmente en: {ruta_salida}")


def main(ruta_config: str = "config.yaml"):
    """Orquesta el pipeline completo de entrenamiento, con tracking en MLflow."""
    config = cargar_config(ruta_config)

    # Configurar MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Cargar datos ya procesados con features (salida del Paso 13)
    df = pd.read_csv(config["paths"]["features_data"])
    X, y = separar_features_target(df, config["target_column"])
    X_train, X_test, y_train, y_test = dividir_train_test(X, y, config)

    with mlflow.start_run(run_name="train_model_pipeline"):
        grid_search = entrenar_con_gridsearch(X_train, y_train, config)
        modelo_final = grid_search.best_estimator_

        metricas = evaluar_modelo(modelo_final, X_test, y_test)

        # Logging en MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("class_weight", config["model"]["class_weight"])
        mlflow.log_metric("recall_cv_promedio", grid_search.best_score_)
        for nombre_metrica, valor in metricas.items():
            mlflow.log_metric(nombre_metrica, valor)
        mlflow.sklearn.log_model(modelo_final, "modelo")
        mlflow.set_tag("estado", "pipeline_automatizado")

        print("Entrenamiento completado.")
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")
        print(f"Métricas en test: {metricas}")

    # Guardamos también una copia local, independiente de MLflow
    guardar_modelo_local(modelo_final, config["paths"]["model_output"])

    return modelo_final, metricas


if __name__ == "__main__":
    main()