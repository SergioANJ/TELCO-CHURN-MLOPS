"""
predict_model.py

Carga el modelo campeón entrenado y genera predicciones de churn sobre
clientes nuevos, aplicando el mismo pipeline de limpieza y encoding
usado en el entrenamiento (make_dataset.py y build_features.py).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

# Permite importar desde src/ sin importarlo como paquete instalado
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.make_dataset import limpiar_datos
from features.build_features import construir_features

def cargar_config(ruta_config: str = "config.yaml") -> dict:
    with open(ruta_config, "r") as f:
        return yaml.safe_load(f)
    
def cargar_modelo(ruta_modelo: str):
    """Carga el modelo entrenado desde disco."""
    modelo = joblib.load(ruta_modelo)
    return modelo

def preparar_datos_nuevos(df_crudo: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de limpieza + encoding usado en entrenamiento,
    sobre datos nuevos que llegan en formato crudo (sin procesar).

    Nota: si df_crudo no trae la columna target (es lo esperable en
    inferencia real), se maneja sin fallar.
    """
    df = df_crudo.copy()

    tiene_target = target_column in df.columns
    if not tiene_target:
        # limpiar_datos() y construir_features() no dependen de Churn,
        # pero si tu pipeline lo requiriera, aquí es donde se ajustaría.
        pass

    df_limpio = limpiar_datos(df)
    df_features = construir_features(df_limpio)

    return df_features

def alinear_columnas(df_nuevo: pd.DataFrame, columnas_entrenamiento: list) -> pd.DataFrame:
    """
    Garantiza que df_nuevo tenga EXACTAMENTE las mismas columnas, en el
    mismo orden, que las usadas en entrenamiento.

    Esto es crítico con One-Hot Encoding: si un cliente nuevo no tiene
    cierta categoría (ej. ningún cliente del lote nuevo usa 'Two year'),
    esa columna no se generaría, y el modelo fallaría o predeciría mal.
    """
    df_alineado = df_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)
    return df_alineado

def predecir(modelo, X_nuevo: pd.DataFrame) -> pd.DataFrame:
    """
    Genera predicciones de clase y probabilidad para cada cliente.

    Returns:
        DataFrame con columnas: prediccion_churn, probabilidad_churn.
    """
    predicciones = modelo.predict(X_nuevo)
    probabilidades = modelo.predict_proba(X_nuevo)[:, 1]

    resultado = pd.DataFrame({
        "prediccion_churn": predicciones,
        "probabilidad_churn": probabilidades
    })
    return resultado

def main(ruta_datos_nuevos: str, ruta_config: str = "config.yaml") -> pd.DataFrame:
    """Orquesta el pipeline completo de inferencia sobre un archivo CSV nuevo."""
    config = cargar_config(ruta_config)

    modelo = cargar_modelo(config["paths"]["model_output"])

    df_crudo = pd.read_csv(ruta_datos_nuevos)
    df_features = preparar_datos_nuevos(df_crudo, config["target_column"])

    # Si el CSV de entrada trae Churn (ej. para validar contra la realidad),
    # lo separamos antes de predecir
    if config["target_column"] in df_features.columns:
        df_features = df_features.drop(columns=[config["target_column"]])

    columnas_entrenamiento = modelo.feature_names_in_.tolist()
    df_alineado = alinear_columnas(df_features, columnas_entrenamiento)

    resultado = predecir(modelo, df_alineado)

    # Anexamos las predicciones a los datos originales, para contexto legible
    resultado_final = pd.concat(
        [df_crudo.reset_index(drop=True), resultado], axis=1
    )

    return resultado_final


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/models/predict_model.py <ruta_al_csv_de_clientes_nuevos>")
        sys.exit(1)

    ruta_entrada = sys.argv[1]
    resultado = main(ruta_entrada)
    print(resultado[["prediccion_churn", "probabilidad_churn"]].head(10))

    ruta_salida = "reports/predicciones.csv"
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(ruta_salida, index=False)
    print(f"\nPredicciones completas guardadas en: {ruta_salida}")