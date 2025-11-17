# Segmentación de Salud Mental con Machine Learning

Este proyecto implementa un modelo de aprendizaje no supervisado (K-Means Clustering) para segmentar perfiles de salud mental y física en grupos de riesgo. Incluye una interfaz web construida con Flask para facilitar el diagnóstico por parte de profesionales médicos.

## 📋 Descripción del Proyecto
El sistema analiza indicadores de salud (tasas de mortalidad por enfermedades crónicas y tasas de suicidio) para clasificar a una región o paciente en uno de dos perfiles:
* **Cluster 0:** Alto Riesgo Fisiológico (Enfermedades Crónicas).
* **Cluster 1:** Riesgo Latente de Salud Mental (Suicidio).

## 🛠️ Requisitos Previos
Para ejecutar este proyecto necesitas tener instalado **Python 3.x**.
Las librerías necesarias se encuentran detalladas en `requirements.txt`.

## 🚀 Guía de Instalación y Puesta en Marcha

### 1. Clonar el repositorio
Descarga el código en tu máquina local.

### 2. Crear y activar un entorno virtual
Es una buena práctica utilizar un entorno virtual para aislar las dependencias. Desde la carpeta raíz del proyecto:

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
Una vez activado el entorno virtual, instala las librerías necesarias:
```bash
pip install -r requirements.txt
```

### 4. Generar el modelo
Antes de iniciar la aplicación, es necesario generar el archivo del modelo (`.joblib`). Si es la primera vez que ejecutas el proyecto, abre y corre todas las celdas del notebook `Clustering_Model.ipynb`.

Esto creará automáticamente el archivo `model/clustering_model.joblib`.

*Nota: Para este paso necesitas tener instalado Jupyter Notebook o Jupyter Lab (`pip install notebook`).*

### 5. Iniciar la Aplicación Web
Ejecuta el servidor de Flask con el siguiente comando:
```bash
python app.py
```
La aplicación estará disponible en `http://127.0.0.1:5000/`.
