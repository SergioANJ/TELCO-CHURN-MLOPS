Cauando estaba construyendo el modelo, en la etapa de EDA, para mirar que variables eran las optimas para pasarle al arbol de decision, se analizarón las variables con el metodo de chi-cuadrado.
Entonces cuando este creando el MLOPS  y arme el pipeline en src/, esta selección de variables por chi-cuadrado se va a convertir en una función feature_selection.py que corre automáticamente — no algo que decidiste "a ojo" una sola vez y ya.

2. Se identificó que la columna TotalCharges tiene 11 nulos (EN el paso 3 debemos mirar como tratarlos)

3. más adelante, cuando construyamos el pipeline en src/, vamos a agregar validaciones automáticas de tipos de dato (con assert o incluso con una librería como pandera o great_expectations)


los .gitkeep se usan para que la carpeta se suba a git pero como se sube vacía esta comando se usa para que git la deje subir como vacía de los contrario si la detecta vacia no la sube entonces se trata de subir para tener una trazabildiad de que ahí van los modelos, y el dataset..

comando para ejecutar mlflow: (venv) PS C:\Users\Usuario\telco-churn-mlops> mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db

# Telco Customer Churn - Predicción con Árbol de Decisión

Proyecto de Machine Learning end-to-end para predecir la fuga de clientes (churn) 
de una empresa de telecomunicaciones, implementado con prácticas de MLOps.

## Objetivo de negocio
Identificar clientes con alta probabilidad de cancelar su servicio, priorizando 
recall (detectar la mayor cantidad posible de clientes en riesgo real de irse), 
dado que el costo de no detectar un cliente que se va es mayor que el de una 
falsa alarma.

## Dataset
Telco Customer Churn (Kaggle) - ~7,043 clientes, 20 variables.

## Estructura del proyecto
telco-churn-mlops/
├── data/
│   ├── raw/                    # Datos originales, sin tocar (ya lo tienes)
│   └── processed/              # Datos limpios (ya lo tienes)
├── notebooks/                  # Tus notebooks exploratorios (ya lo tienes)
├── models/                     # Modelos entrenados serializados (.pkl)
├── src/                        # Código fuente modularizado
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py     # Función que hace la limpieza (Paso 3, ahora como script)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Encoding, feature engineering (Paso 3.5)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py      # Entrenamiento (Paso 5-9)
│   │   └── predict_model.py    # Inferencia con el modelo campeón
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py        # Gráficos del EDA (Paso 2)
├── mlruns/ o mlflow.db          # Tracking de MLflow (dedicado a este proyecto)
├── reports/
│   └── figures/                 # Gráficos exportados
├── requirements.txt              # Dependencias del proyecto
├── config.yaml                   # Parámetros configurables (hiperparámetros, rutas)
├── .gitignore
└── README.md

## Modelo final
Árbol de Decisión con `class_weight='balanced'`, seleccionado por mejor 
recall (0.805) y F1-score entre 4 experimentos comparados en MLflow 
(baseline, podado, class_weight, SMOTE).

## Cómo reproducir
1. Clonar el repo
2. `pip install -r requirements.txt`
3. Colocar el dataset en `data/raw/`
4. (instrucciones de ejecución, las completamos en los siguientes pasos)

## Autor
[Tu nombre] - Proyecto para Diplomado de IA / Especialización en MLOps