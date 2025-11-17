from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
try:
    # Asegúrate de que el archivo joblib esté en la carpeta 'model'
    model = joblib.load('model/clustering_model.joblib')
except FileNotFoundError:
    model = None

# Definir las interpretaciones médicas basadas en el análisis de centroides
interpretations = {
    0: {
        "title": "Grupo de Carga de Enfermedad Crónica",
        "risk_level": "Alto Riesgo Fisiológico / Riesgo Medio Suicida",
        "desc": "Este paciente/región pertenece a un grupo con alta probabilidad de muerte prematura por enfermedades cardiovasculares, cáncer o diabetes (~25%).",
        "action": "Priorizar control de enfermedades crónicas. Mantener vigilancia estándar en salud mental."
    },
    1: {
        "title": "Grupo de Riesgo de Salud Mental Latente",
        "risk_level": "Bajo Riesgo Fisiológico / Riesgo Aumentado Suicida",
        "desc": "Este perfil presenta buena salud física general, pero estadísticas de suicidio superiores a la media del otro grupo (~13 por 100k).",
        "action": "ALERTA: Enfocar recursos en prevención del suicidio y bienestar psicológico. El buen estado físico puede enmascarar cuadros depresivos."
    }
}

# Columnas requeridas por el modelo (Basado en tu X_train)
model_columns = [
    'unnamed:_1', # Año/Índice temporal
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease',
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease.1',
    'probability_(%)_of_dying_between_age_30_and_exact_age_70_from_any_of_cardiovascular_disease,_cancer,_diabetes,_or_chronic_respiratory_disease.2',
    'crude_suicide_rates_(per_100_000_population).1',
    'crude_suicide_rates_(per_100_000_population).2'
]

@app.route('/')
def home():
    # Renderiza el formulario HTML
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Modelo no encontrado. Ejecuta el notebook para crear el archivo .joblib", 500

    try:
        # Recolectar datos
        input_data = []
        for col in model_columns:
            val = request.form.get(col)
            if val is None:
                return f"Falta el valor para: {col}", 400
            input_data.append(float(val))
        
        # Crear DataFrame
        input_df = pd.DataFrame([input_data], columns=model_columns)
        
        # Predecir
        cluster = int(model.predict(input_df)[0])
        
        # Obtener interpretación
        result_info = interpretations.get(cluster, {"title": "Desconocido", "desc": "Sin datos"})
        
        return render_template(
            'result.html', 
            cluster=cluster, 
            title=result_info['title'],
            risk=result_info['risk_level'],
            description=result_info['desc'],
            action=result_info['action']
        )

    except Exception as e:
        return f"Error en la predicción: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)