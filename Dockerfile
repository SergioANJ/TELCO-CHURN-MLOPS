# -- Imagen base ---
# Python 3.11 en su versión "slim": Linux minimalista, sin herramientas
# innecesarias, para que la imagen final sea más liviana y rápida de construir
FROM python:3.12-slim

#--- Directorio de trabajo dentro del contenedor ---
# Todo lo que hagamos de aquí en adelante ocurre dentro de /app,
# dentro del contenedor (es una carpeta nueva, no tiene relación con la carpeta local del mismo nombre)
WORKDIR /app

# ---- Instalación de dependencias ----
# Copiamos SOLO requirements.txt primero (no todo el código todavía).
# Esto es una técnica de optimización: Docker cachea cada paso (layer).
# Si el código cambia pero requirements.txt no, Docker reutiliza la capa
# ya instalada en vez de reinstalar todo de cero cada vez.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ---- Copiamos el código y artefactos necesarios ----
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .

# ---- Puerto que expone el contenedor ----
# Uvicorn va a escuchar en el puerto 8000 dentro del contenedor
EXPOSE 8000

# ---- Comando que se ejecuta al iniciar el contenedor ----
# Ojo: usamos --host 0.0.0.0 (no 127.0.0.1) para que la API sea
# accesible DESDE FUERA del contenedor, no solo desde dentro
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]