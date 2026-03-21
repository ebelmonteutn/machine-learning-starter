# predict.py
# Este script se encarga de cargar un modelo previamente entrenado y realizar predicciones sobre nuevos datos.
# Autor: Enzo Nicolas Belmonte
# Fecha: 2024-03-21
# Licencia: MIT License. Ver LICENSE para más detalles.
import joblib
import pandas as pd


def load_model(path):
    """
    Carga un modelo previamente entrenado desde disco.
    """
    return joblib.load(path)


def predict(model, input_data):
    """
    Realiza predicciones sobre nuevos datos.

    Parámetros:
    - model: modelo entrenado
    - input_data: DataFrame con mismo formato que entrenamiento

    Retorna:
    - predicciones
    """
    return model.predict(input_data)


def main():
    """
    Simula un caso real de inferencia (producción):
    - Se carga el modelo
    - Se reciben nuevos datos
    - Se devuelve la predicción
    """

    model = load_model("models/model.joblib")

    # IMPORTANTE:
    # Las columnas deben coincidir EXACTAMENTE con las usadas en entrenamiento
    new_data = pd.DataFrame([
        {"feature1": 1.2, "feature2": 3.4}
    ])

    prediction = predict(model, new_data)

    print("--------------------------|--------------------------")
    print("Prediction:", prediction)
    print("--------------------------|--------------------------")


if __name__ == "__main__":
    main()