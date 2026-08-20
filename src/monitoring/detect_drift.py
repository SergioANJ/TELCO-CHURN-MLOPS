"""
detect_drift.py

Detecta data drift comparando el dataset de entrenamiento (referencia/baseline)
contra un lote de datos nuevos, usando Evidently AI.

Evidently elige automáticamente el test estadístico apropiado por columna
(Kolmogorov-Smirnov o Wasserstein para numéricas, Chi-cuadrado o
Jensen-Shannon para categóricas, según el tamaño de la muestra).
"""

import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.make_dataset import limpiar_datos


def generar_reporte_drift(
    ruta_baseline: str,
    ruta_nuevo: str,
    ruta_salida_html: str
) -> Report:
    """
    Compara el dataset de referencia (entrenamiento) contra datos nuevos,
    y genera un reporte HTML interactivo con el análisis de drift.
    """
    df_baseline = limpiar_datos(pd.read_csv(ruta_baseline))
    df_nuevo = limpiar_datos(pd.read_csv(ruta_nuevo))

    # customerID no aporta como feature (ya lo sabíamos desde el EDA),
    # y limpiar_datos() ya lo elimina, así que ambos datasets llegan
    # limpios y comparables

    reporte = Report([DataDriftPreset()])
    resultado = reporte.run(current_data=df_nuevo, reference_data=df_baseline)

    Path(ruta_salida_html).parent.mkdir(parents=True, exist_ok=True)
    resultado.save_html(ruta_salida_html)
    print(f"Reporte HTML de drift guardado en: {ruta_salida_html}")

    return resultado


def resumen_en_consola(resultado: Report) -> None:
    """Imprime un resumen legible directamente en la terminal."""
    resultado_dict = resultado.dict()
    print("\nResumen del análisis de drift generado correctamente.")
    print(f"Abre el archivo HTML para ver el reporte visual completo.")


if __name__ == "__main__":
    RUTA_BASELINE = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    RUTA_NUEVO = "data/monitoring/datos_nuevos_simulados.csv"
    RUTA_SALIDA_HTML = "reports/drift_report.html"

    resultado = generar_reporte_drift(RUTA_BASELINE, RUTA_NUEVO, RUTA_SALIDA_HTML)
    resumen_en_consola(resultado)