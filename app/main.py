"""
main.py

API REST para predicción de churn, construida con FastAPI.
Reutiliza el mismo pipeline de limpieza/encoding usado en entrenamiento,
para garantizar consistencia total entre entrenamiento e inferencia.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import limpiar_datos
from features.build_features import construir_features
from models.predict_model import alinear_columnas


# --- Definición del "contrato" de datos de entrada ---

class ClienteInput(BaseModel):
    """
    Representa un cliente en su forma cruda, tal como llega del negocio
    (sin codificar), igual a las columnas del CSV original (sin Churn).
    """
    customerID: str = Field(..., example="7590-VHVEG")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: str = Field(..., example="29.85")


class PrediccionOutput(BaseModel):
    """Respuesta de la API: predicción y probabilidad."""
    customerID: str
    prediccion_churn: int
    probabilidad_churn: float
    interpretacion: str


# --- Carga del modelo (una sola vez, al iniciar la API) ---

def cargar_config(ruta_config: str = "config.yaml") -> dict:
    with open(ruta_config, "r") as f:
        return yaml.safe_load(f)


config = cargar_config()
modelo = joblib.load(config["paths"]["model_output"])
columnas_entrenamiento = modelo.feature_names_in_.tolist()


# --- Definición de la API ---

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir la probabilidad de fuga (churn) de clientes",
    version="1.0.0"
)


@app.get("/")
def inicio():
    """Endpoint raíz, útil para confirmar que la API está viva."""
    return {"mensaje": "Telco Churn Prediction API - usa /predict para predecir", "status": "ok"}


@app.get("/health")
def health_check():
    """Endpoint de salud, típico en APIs de producción para monitoreo automático."""
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.post("/predict", response_model=PrediccionOutput)
def predecir_cliente(cliente: ClienteInput):
    """
    Recibe un cliente en formato crudo y devuelve la predicción de churn.
    """
    try:
        df_crudo = pd.DataFrame([cliente.dict()])

        df_limpio = limpiar_datos(df_crudo)
        df_features = construir_features(df_limpio)
        df_alineado = alinear_columnas(df_features, columnas_entrenamiento)

        prediccion = int(modelo.predict(df_alineado)[0])
        probabilidad = float(modelo.predict_proba(df_alineado)[:, 1][0])

        interpretacion = (
            "Alto riesgo de fuga, se recomienda acción de retención"
            if prediccion == 1
            else "Bajo riesgo de fuga"
        )

        return PrediccionOutput(
            customerID=cliente.customerID,
            prediccion_churn=prediccion,
            probabilidad_churn=round(probabilidad, 4),
            interpretacion=interpretacion
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")