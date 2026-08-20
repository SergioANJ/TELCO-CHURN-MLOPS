

# Telco Customer Churn — Proyecto MLOps End-to-End

Proyecto de Machine Learning completo, desde la exploración de datos hasta el despliegue, implementando prácticas reales de MLOps. Desarrollado como práctica integradora de un diplomado de Inteligencia Artificial y una especialización en MLOps.

## Objetivo de negocio

Predecir qué clientes de una empresa de telecomunicaciones tienen alta probabilidad de cancelar su servicio (churn), para permitir acciones de retención proactivas. El modelo prioriza **recall** sobre precision: el costo de no detectar a un cliente que se va (falso negativo) es mayor que el costo de una falsa alarma (falso positivo).

## Dataset

**Telco Customer Churn** — ~7,043 clientes, 20 variables (demográficas, de servicios contratados y de facturación). Variable objetivo: `Churn` (Yes/No), con un desbalance de clases de aproximadamente 73%/27%.

---

## Tabla de contenidos

1. [Análisis Exploratorio de Datos (EDA)](#1-análisis-exploratorio-de-datos-eda)
2. [Limpieza y Preprocesamiento](#2-limpieza-y-preprocesamiento)
3. [Feature Engineering](#3-feature-engineering)
4. [Modelado y Experimentación](#4-modelado-y-experimentación)
5. [Tracking de Experimentos con MLflow](#5-tracking-de-experimentos-con-mlflow)
6. [Arquitectura del Proyecto (Cookiecutter Data Science)](#6-arquitectura-del-proyecto)
7. [Testing Automatizado](#7-testing-automatizado)
8. [Control de Versiones de Datos con DVC](#8-control-de-versiones-de-datos-con-dvc)
9. [API de Inferencia con FastAPI](#9-api-de-inferencia-con-fastapi)
10. [Containerización con Docker](#10-containerización-con-docker)
11. [Orquestación con Docker Compose](#11-orquestación-con-docker-compose)
12. [CI/CD con GitHub Actions](#12-cicd-con-github-actions)
13. [Monitoreo y Detección de Data Drift](#13-monitoreo-y-detección-de-data-drift)
14. [Cómo reproducir este proyecto](#14-cómo-reproducir-este-proyecto)
15. [Estructura de carpetas](#15-estructura-de-carpetas)

---

## 1. Análisis Exploratorio de Datos (EDA)

Ubicación: `notebooks/01_eda.ipynb`

El EDA se realizó con rigor estadístico, evitando decisiones "a ojo":

- **Inspección de tipos de dato**: se detectó que `TotalCharges` estaba mal tipada como texto (`object`), con nulos "disfrazados" como strings vacíos (`" "`), en vez de `NaN` explícitos.
- **Análisis de la variable objetivo**: se identificó el desbalance de clases (~73% No / ~27% Yes), lo cual determinó decisiones posteriores de modelado (métrica de evaluación, necesidad de balanceo).
- **Análisis univariado y bivariado**: distribución de cada variable, y su relación con `Churn`.
- **Selección de variables con evidencia estadística, no intuición**:
  - Variables **categóricas** evaluadas con **test de Chi-cuadrado** contra `Churn`. `Contract` resultó la variable más asociada (chi2 más alto, p-valor ≈ 0), mientras `gender` y `PhoneService` no mostraron asociación significativa.
  - Variables **numéricas** evaluadas con **Mann-Whitney U** (no paramétrico, apropiado dado que las distribuciones no son necesariamente normales). `tenure` resultó la más asociada al churn.
- **Verificación de multicolinealidad**: se detectó una correlación de 0.826 entre `tenure` y `TotalCharges` — documentada pero no tratada, dado que el modelo elegido (árbol de decisión) es robusto a la colinealidad entre features.

## 2. Limpieza y Preprocesamiento

Cada decisión de limpieza fue tomada con base en evidencia, no por defecto:

- **`customerID`**: eliminado (identificador único, no predictivo).
- **`TotalCharges`**: convertido a numérico. Los 11 valores nulos correspondían exclusivamente a clientes con `tenure=0` (clientes nuevos sin historial de facturación) — se imputaron con 0, ya que representa el valor real de facturación acumulada, no una estimación estadística.
- **Outliers**: revisados visualmente (boxplots); se determinó que son valores legítimos de negocio (clientes con alto consumo), no errores de captura. No se aplicó tratamiento, dado que además el modelo elegido es robusto a valores extremos.
- **Duplicados**: se identificaron 42 filas involucradas en duplicados exactos (tras eliminar `customerID`). Se decidió **conservarlos**, ya que 24 de esos 42 correspondían a `Churn=Yes` — eliminarlos habría agravado el desbalance de clases ya existente.

Implementado en: `src/data/make_dataset.py`, con validaciones automáticas (`assert`) que verifican que las premisas de limpieza (ej. "todos los nulos de TotalCharges corresponden a tenure=0") se sigan cumpliendo ante datos nuevos.

## 3. Feature Engineering

Estrategia de encoding diferenciada según el tipo de variable:

- **Variables binarias** (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`): mapeo manual 0/1.
- **`Contract`**: **Ordinal Encoding** (Month-to-month=0, One year=1, Two year=2), porque existe una jerarquía real de duración/compromiso — permite al árbol hacer splits del tipo "¿Contract ≥ 1 año?" en un solo corte.
- **Resto de categóricas nominales** (`InternetService`, `PaymentMethod`, etc.): **One-Hot Encoding** con `drop_first=True`, evitando la "trampa de la variable dummy" (multicolinealidad perfecta).

Implementado en: `src/features/build_features.py`, con validaciones que detectan valores categóricos inesperados antes de que corrompan silenciosamente el pipeline.

## 4. Modelado y Experimentación

Se compar944 objetivamente **4 configuraciones** de un árbol de decisión, todas con búsqueda de hiperparámetros vía `GridSearchCV` (5-fold cross-validation, optimizando `recall`):

| Experimento | Recall (test) | Precision (test) | F1 |
|---|---|---|---|
| Baseline (sin límites, overfitting: train 99.8% / test 73.7% accuracy) | 0.497 | — | — |
| Podado (GridSearch de `max_depth`/`min_samples_split`/`leaf`) | 0.583 | — | — |
| Podado + `class_weight='balanced'` | **0.805** | 0.528 | **0.637** |
| Podado + SMOTE (oversampling sintético) | 0.663 | 0.537 | 0.594 |

**Modelo campeón**: Podado + `class_weight='balanced'`, elegido por mejor recall y F1-score, alineado con el criterio de negocio de priorizar la detección de clientes en riesgo de fuga.

## 5. Tracking de Experimentos con MLflow

Todos los experimentos (parámetros, métricas, y el modelo serializado) quedaron trackeados en MLflow, con backend en SQLite (`mlflow.db`), permitiendo comparar resultados de forma objetiva en vez de "a mano" en el notebook.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 6. Arquitectura del Proyecto

Estructura basada en **Cookiecutter Data Science**, separando claramente:
- `notebooks/` — exploración humana (EDA).
- `src/` — código de producción, modularizado y reutilizable (`data/`, `features/`, `models/`, `monitoring/`).
- `data/raw/` vs `data/processed/` — datos originales intocados vs. datos transformados.
- Cada función de `src/` sigue el principio de responsabilidad única, con `assert` de validación de datos que convierten supuestos verificados manualmente durante el EDA en chequeos automáticos permanentes.

## 7. Testing Automatizado

18 tests con `pytest`, cubriendo:
- Limpieza de datos (`tests/test_make_dataset.py`), incluyendo un test que confirma que el pipeline **falla correctamente** ante datos que violan la premisa de negocio esperada.
- Feature engineering (`tests/test_build_features.py`).
- Pipeline de entrenamiento (`tests/test_train_model.py`) — solo funciones rápidas, sin reentrenar el modelo completo en cada corrida.
- Pipeline de inferencia (`tests/test_predict_model.py`), usando un modelo "mock" (test double) para probar la lógica de alineación de columnas sin depender de un modelo real.

```bash
pytest tests/ -v
```

## 8. Control de Versiones de Datos con DVC

El dataset crudo, el dataset de features y el modelo entrenado están versionados con **DVC**, con **Google Drive** como remote de almacenamiento. La autenticación usa una **cuenta de servicio** de Google Cloud (necesaria para autenticación no interactiva en CI/CD), con las credenciales gestionadas de forma segura (nunca committeadas — se detectó y corrigió un incidente real de exposición de credenciales OAuth durante el desarrollo, resuelto moviéndolas a `.dvc/config.local`, excluido de Git).

```bash
dvc pull   # descarga dataset y modelo versionados
```

## 9. API de Inferencia con FastAPI

`app/main.py` expone el modelo como servicio REST, reutilizando el mismo pipeline de limpieza/encoding usado en entrenamiento (garantizando consistencia total entre entrenamiento e inferencia). Recibe clientes en su forma **cruda** (sin codificar), como los conocería cualquier sistema de negocio externo (CRM, formulario web).

- `GET /health` — healthcheck.
- `POST /predict` — recibe un cliente, devuelve predicción + probabilidad de churn + interpretación.
- Documentación interactiva autogenerada: `/docs` (Swagger UI).

Incluye alineación robusta de columnas (`alinear_columnas`) para evitar el error silencioso más común en producción con One-Hot Encoding: que un lote de inferencia no contenga todas las categorías vistas en entrenamiento.

## 10. Containerización con Docker

La API está empaquetada en una imagen Docker (`Dockerfile`), con un `requirements-api.txt` curado específicamente para producción — excluyendo dependencias de desarrollo (Jupyter, DVC, MLflow, pytest) que no son necesarias para servir el modelo, reduciendo superficie de error y tamaño de imagen.

```bash
docker build -t telco-churn-api .
docker run -p 8080:8000 telco-churn-api
```

## 11. Orquestación con Docker Compose

`docker-compose.yml` levanta dos servicios coordinados:
- **`api`**: la API de FastAPI.
- **`mlflow`**: servidor de MLflow en su propio contenedor, con un volumen persistente (`mlflow-data`) para que el historial de experimentos sobreviva a reinicios del contenedor.

```bash
docker-compose up --build
```

## 12. CI/CD con GitHub Actions

`.github/workflows/tests.yml` automatiza, en cada `push` a `main`:

1. **Job `test`**: instala dependencias en un entorno limpio (Ubuntu) y corre los 18 tests.
2. **Job `docker-build`** (depende de que `test` pase): descarga el modelo vía `dvc pull` (autenticado con cuenta de servicio), construye la imagen Docker, la levanta, y verifica con `curl` que el endpoint `/health` responde correctamente — validando de punta a punta que un despliegue real funcionaría.

## 13. Monitoreo y Detección de Data Drift

`src/monitoring/` implementa detección de **data drift** (cambios en la distribución de las variables de entrada a lo largo del tiempo) usando **Evidently AI**, comparando el dataset de entrenamiento (referencia) contra un lote de "datos nuevos".

- `simulate_drift.py`: genera un dataset sintético con drift artificial controlado (cambio en `Contract` y `MonthlyCharges`), útil para validar que el detector funciona correctamente.
- `detect_drift.py`: genera un reporte HTML interactivo (`reports/drift_report.html`) con el análisis de drift por columna.

```bash
python src/monitoring/simulate_drift.py
python src/monitoring/detect_drift.py
```

## 14. Cómo reproducir este proyecto

```bash
git clone https://github.com/SergioANJ/TELCO-CHURN-MLOPS.git
cd TELCO-CHURN-MLOPS
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
dvc pull                       # descarga datos y modelo (requiere autenticación con Google Drive)
pytest tests/ -v                # verificar que todo funciona
python src/models/train_model.py   # reentrenar si se desea
uvicorn app.main:app --reload      # levantar la API localmente
```

## 15. Estructura de carpetas

```text
telco-churn-mlops/
├── .github/workflows/ # CI/CD
├── app/ # API FastAPI
├── data/
│ ├── raw/ # Datos originales (versionados con DVC)
│ ├── processed/ # Datos limpios y features (versionados con DVC)
│ └── monitoring/ # Datos simulados para pruebas de drift
├── models/ # Modelo entrenado (versionado con DVC)
├── notebooks/ # EDA exploratorio
├── reports/ # Reportes generados (predicciones, drift)
├── src/
│ ├── data/ # Limpieza de datos
│ ├── features/ # Feature engineering
│ ├── models/ # Entrenamiento e inferencia
│ └── monitoring/ # Detección de drift
├── tests/ # 18 tests automatizados
├── config.yaml # Configuración centralizada
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.mlflow
└── requirements.txt
```

---

## Stack tecnológico

`Python` · `pandas` · `scikit-learn` · `scipy` · `imbalanced-learn` · `MLflow` · `DVC` · `FastAPI` · `Docker` · `Docker Compose` · `GitHub Actions` · `Evidently AI` · `pytest`

## Autor

Sergio — Proyecto desarrollado como práctica integradora del Diplomado en Inteligencia Artificial y Especialización en MLOps.