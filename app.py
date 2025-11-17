from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo
try:
    model = joblib.load('model/clustering_model.joblib')
except FileNotFoundError:
    print("Error: El archivo 'model/clustering_model.joblib' no fue encontrado.")
    print("Por favor, ejecuta el notebook 'Clustering_Model.ipynb' para generar el modelo.")
    model = None

# Definir las interpretaciones de los clusters
interpretations = {
    0: "Este perfil corresponde a un grupo con una tasa de suicidios SUPERIOR a la media. Se considera de ALTO RIESGO y requiere atención prioritaria.",
    1: "Este perfil corresponde a un grupo con una tasa de suicidios INFERIOR a la media. Se considera de BAJO RIESGO, aunque no se debe descartar la vigilancia."
}

# Nombres de las columnas como se esperan en el modelo
# Extraídos del notebook, después de cargar X_train con index_col=0
model_columns = [
    'unnamed:_1',
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease',
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease.1',
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease.2',
    'crude_suicide_rates_(per_100_000_population).1',
    'crude_suicide_rates_(per_100_000_population).2'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: El modelo no está cargado. Ejecuta el notebook para generarlo.", 500

    try:
        # Recolectar los datos del formulario
        form_data = request.form.to_dict()
        
        # Convertir los datos a un array de numpy en el orden correcto
        input_data = [float(form_data[col]) for col in model_columns]
        
        # Crear un DataFrame para la predicción
        input_df = pd.DataFrame([input_data], columns=model_columns)
        
        # Realizar la predicción
        prediction = model.predict(input_df)
        cluster = prediction[0]
        
        # Obtener la interpretación
        interpretation = interpretations.get(cluster, "No se encontró una interpretación para este cluster.")
        
        return render_template('result.html', cluster=cluster, interpretation=interpretation)

    except Exception as e:
        return f"Ocurrió un error durante la predicción: {e}", 400

if __name__ == '__main__':
    # Verificar si el modelo existe antes de correr la app
    try:
        joblib.load('model/clustering_model.joblib')
        print("✓ Modelo encontrado. Iniciando la aplicación Flask...")
        app.run(debug=True)
    except FileNotFoundError:
        print("✗ Error Crítico: No se puede iniciar la aplicación porque el modelo 'model/clustering_model.joblib' no existe.")
        print("→ Por favor, abre y ejecuta todas las celdas del notebook 'Clustering_Model.ipynb' para crearlo.")

