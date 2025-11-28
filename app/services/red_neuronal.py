import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
from typing import Dict, Any, List
from app.models.registro_ml import SesionDTO, HorarioItem
import datetime

MODEL_PATH = "data/modelo.pkl"

# ---------------------------
# Balanceo del dataset
# ---------------------------
def balancear_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    df["hora"] = df["hora"].apply(convertir_hora)

    # Separar por clase
    clases = []
    for estado in [1,2,3,4,5]:
        clase = df[df["estadoAnimo"] == estado]
        if not clase.empty:
            clases.append(clase)

    # Encontrar tamaño máximo
    max_size = max(len(c) for c in clases)

    # Oversampling: duplicar clases minoritarias hasta igualar
    clases_balanceadas = [
        resample(c, replace=True, n_samples=max_size, random_state=42)
        for c in clases
    ]

    df_balanceado = pd.concat(clases_balanceadas)
    return df_balanceado.sample(frac=1, random_state=42)  # mezclar

# ---------------------------
# Entrenamiento completo balanceado
# ---------------------------
def entrenar_completo(csv_path: str):
    df = balancear_dataset(csv_path)
    X = df[["duracion","tecnica","hora","dia"]]
    y = df["estadoAnimo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=500, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

# ---------------------------
# Entrenamiento incremental con buffer
# ---------------------------
buffer = []

def entrenar_incremental(X, y):
    global buffer
    buffer.extend(list(zip(X, y)))

    # Solo entrenar si hay variedad de estados
    estados_unicos = set(lbl for _, lbl in buffer)
    if len(estados_unicos) < 2:
        print("Esperando más variedad antes de entrenar incremental...")
        return

    X_batch, y_batch = zip(*buffer)
    buffer = []  # limpiar

    try:
        model = joblib.load(MODEL_PATH)
        model.partial_fit(X_batch, y_batch)
    except:
        model = SGDClassifier(loss="log_loss", class_weight="balanced")
        model.partial_fit(X_batch, y_batch, classes=[1,2,3,4,5])

    joblib.dump(model, MODEL_PATH)

# ---------------------------
# Predicción
# ---------------------------
def predecir_estado(registro: Dict[str, Any]) -> int:
    model = joblib.load(MODEL_PATH)
    hora_decimal = convertir_hora(registro["hora"]) if isinstance(registro["hora"], str) else registro["hora"]
    X = [[registro["duracion"], registro["tecnica"], hora_decimal, registro["dia"]]]
    return int(model.predict(X)[0])

# ---------------------------
# Mejores horarios
# ---------------------------
def mejores_horarios(csv_path: str, top_n: int = 3) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, names=["duracion","tecnica","estadoAnimo","hora","dia"])
    df["hora"] = df["hora"].apply(convertir_hora)
    horarios = df.groupby("hora")["estadoAnimo"].mean().reset_index()
    mejores = horarios.sort_values(by="estadoAnimo", ascending=False).head(top_n)
    return mejores.to_dict(orient="records")

# ---------------------------
# Calcular horarios
# ---------------------------
def calcular_horarios(sesiones: List[SesionDTO]) -> List[HorarioItem]:
    print("Sesiones recibidas:", sesiones)
    horarios: List[HorarioItem] = []

    if not sesiones:
        return [HorarioItem(hora=0.5, estadoAnimo=3)]  # 12:00 como fracción (0–1)

    # Agregados del historial
    duracion_promedio = max(60, int(sum(s.duracionSegundos for s in sesiones) / len(sesiones)))
    tecnica_mas_usada = max(set(s.rutinaId for s in sesiones),
                            key=lambda t: sum(1 for s in sesiones if s.rutinaId == t))
    dia_actual = datetime.datetime.now().weekday()

    horas_fraccion = [0.08, 0.25, 0.5, 0.75, 0.85, 0.99]

    for fr in horas_fraccion:
        try:
            estado_predicho = predecir_estado({
                "duracion": duracion_promedio,
                "tecnica": tecnica_mas_usada,
                "hora": fr * 24,
                "dia": dia_actual
            })
            print(f"[ML] hora(fr)={fr:.2f} (dec={fr*24:.2f}) -> predicho={estado_predicho}")
            horarios.append(HorarioItem(hora=fr, estadoAnimo=estado_predicho))
        except Exception as e:
            print(f"Error en predicción para hora {fr}: {e}")

    if len(horarios) < 3:
        extras = [h for h in [0.25, 0.5, 0.75] if h not in [it.hora for it in horarios]]
        for h in extras:
            horarios.append(HorarioItem(hora=h, estadoAnimo=3))

    return horarios

# ---------------------------
# Conversión de hora y día
# ---------------------------
def convertir_hora(hora_str: str) -> float:
    h, m = map(int, hora_str.split(":"))
    return h + m/60.0

def convertir_dia(fecha_str: str) -> int:
    fecha = datetime.datetime.strptime(fecha_str, "%Y-%m-%d")
    return fecha.weekday()