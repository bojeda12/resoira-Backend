import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import Counter
from typing import Dict, Any, List
from app.models.registro_ml import SesionDTO, HorarioItem





MODEL_PATH = "data/modelo.pkl"

def entrenar_modelo(csv_path: str):
    """
    Entrena un modelo de red neuronal MLP usando los datos del CSV
    y guarda el modelo entrenado en disco.
    """
    # Cargar dataset
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    
    # Features y target
    X = df[["duracion","tecnica","hora","dia"]]
    y = df["estadoAnimo"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Definir y entrenar modelo
    model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=500)
    model.fit(X_train, y_train)

    # Guardar modelo entrenado
    joblib.dump(model, MODEL_PATH)


def predecir_estado(registro: Dict[str, Any]) -> int:
    """
    Carga el modelo entrenado y predice el estado de ánimo
    para un registro individual.
    """
    model = joblib.load(MODEL_PATH)
    X = [[registro["duracion"], registro["tecnica"], registro["hora"], registro["dia"]]]
    return int(model.predict(X)[0])


def mejores_horarios(csv_path: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Calcula los mejores horarios basados en el promedio de estado de ánimo
    agrupado por hora en el dataset.
    """
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    
    # Agrupar por hora y calcular promedio de estadoAnimo
    horarios = df.groupby("hora")["estadoAnimo"].mean().reset_index()
    
    # Ordenar por mejor estadoAnimo promedio
    mejores = horarios.sort_values(by="estadoAnimo", ascending=False).head(top_n)
    
    return mejores.to_dict(orient="records")

def calcular_horarios(sesiones: List[SesionDTO]) -> List[HorarioItem]:
    """
    Genera recomendaciones de horarios en base a las sesiones registradas.
    Usa la duración y el estado de ánimo como factores simples.
    """
    horarios = []

    for sesion in sesiones:
        try:
            # Convertir hora "HH:MM" a fracción del día (0.0 - 1.0)
            partes = sesion.horaDelDia.split(":")
            hora = int(partes[0])
            minuto = int(partes[1])
            fraccion_dia = (hora + minuto / 60) / 24.0

            # Si la sesión fue larga y el estado fue positivo, se considera buen horario
            if sesion.duracionSegundos > 5 and sesion.estadoAnimo >= 3:
                horarios.append(HorarioItem(hora=fraccion_dia, estadoAnimo=sesion.estadoAnimo))

        except Exception as e:
            print(f"Error procesando sesión {sesion}: {e}")

    # Si no se encontró nada, devolver al menos un horario por defecto
    if not horarios:
        horarios.append(HorarioItem(hora=0.5, estadoAnimo=3))

    return horarios

