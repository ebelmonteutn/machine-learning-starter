import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
import joblib


# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================

RANDOM_STATE = 42      # Semilla para reproducibilidad (clave en ML)
TEST_SIZE = 0.2        # 20% de los datos para testing


# ==============================
# CARGA DE DATOS
# ==============================

def load_data(path):
    """
    Carga el dataset desde un archivo CSV.

    Parámetros:
    - path: ruta al archivo

    Retorna:
    - DataFrame de pandas
    """
    df = pd.read_csv(path)
    return df


# ==============================
# PREPROCESAMIENTO
# ==============================

def preprocess(df, target_column):
    """
    Separa variables independientes (X) de la variable objetivo (y).

    Parámetros:
    - df: dataset completo
    - target_column: nombre de la variable a predecir

    Retorna:
    - X: features (entrada del modelo)
    - y: labels (salida esperada)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# ==============================
# PIPELINE DE MACHINE LEARNING
# ==============================

def build_pipeline():
    """
    Construye un pipeline de ML que automatiza:
    1. Imputación de valores faltantes
    2. Normalización de datos
    3. Entrenamiento del modelo

    Ventaja clave:
    - Evita data leakage
    - Hace el flujo reproducible y escalable
    """

    pipeline = Pipeline([
        # Reemplaza valores faltantes por la media de cada columna
        ("imputer", SimpleImputer(strategy="mean")),

        # Escala los datos (media=0, varianza=1)
        # Importante para muchos modelos (aunque RandomForest no lo necesita tanto)
        ("scaler", StandardScaler()),

        # Modelo de ML (puede cambiarse sin afectar el resto del pipeline)
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    return pipeline


# ==============================
# ENTRENAMIENTO
# ==============================

def train(X_train, y_train):
    """
    Entrena el modelo con los datos de entrenamiento.

    Parámetros:
    - X_train: features de entrenamiento
    - y_train: etiquetas de entrenamiento

    Retorna:
    - modelo entrenado
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


# ==============================
# EVALUACIÓN
# ==============================

def evaluate(model, X_test, y_test):
    """
    Evalúa el modelo con datos que no vio durante el entrenamiento.

    Métricas:
    - Accuracy: proporción de aciertos
    - Classification report: precision, recall, f1-score
    """
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# ==============================
# PERSISTENCIA
# ==============================

def save_model(model, path):
    """
    Guarda el modelo entrenado en disco.

    Esto permite:
    - No reentrenar cada vez
    - Usarlo en producción / APIs

    Formato: joblib (eficiente para objetos grandes)
    """
    joblib.dump(model, path)


# ==============================
# PIPELINE COMPLETO (ENTRYPOINT)
# ==============================

def main():
    """
    Orquesta todo el flujo de ML:
    1. Carga datos
    2. Preprocesa
    3. Divide dataset
    4. Entrena modelo
    5. Evalúa
    6. Guarda modelo
    """

    # 1. Carga
    df = load_data("data/raw.csv")

    # 2. Separación de variables
    X, y = preprocess(df, target_column="target")

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 4. Entrenamiento
    model = train(X_train, y_train)

    # 5. Evaluación
    evaluate(model, X_test, y_test)

    # 6. Guardado
    save_model(model, "models/model.joblib")


if __name__ == "__main__":
    main()