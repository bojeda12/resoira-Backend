import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import Counter

MODEL_PATH = "data/modelo.pkl"

def entrenar_modelo(csv_path: str):
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    X = df[["duracion","tecnica","hora","dia"]]
    y = df["estadoAnimo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=500)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

def predecir_estado(registro: dict):
    model = joblib.load(MODEL_PATH)
    X = [[registro["duracion"], registro["tecnica"], registro["hora"], registro["dia"]]]
    return model.predict(X)[0]

def mejores_horarios(csv_path: str, top_n: int = 3):
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    # Agrupar por hora y calcular promedio de estadoAnimo
    horarios = df.groupby("hora")["estadoAnimo"].mean().reset_index()
    # Ordenar por mejor estadoAnimo promedio
    mejores = horarios.sort_values(by="estadoAnimo", ascending=False).head(top_n)
    return mejores.to_dict(orient="records")