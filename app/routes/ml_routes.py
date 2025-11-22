from fastapi import APIRouter
from typing import List
import pandas as pd
from app.models.registro_ml import RegistroML
from app.services.red_neuronal import entrenar_modelo, predecir_estado, mejores_horarios

router = APIRouter()

@router.post("/datos")
def recibir_datos(registros: List[RegistroML]):
    df = pd.DataFrame([r.dict() for r in registros])
    df.to_csv("data/datos.csv", mode="a", header=False, index=False)
    entrenar_modelo("data/datos.csv")
    return {"mensaje": "Datos recibidos y modelo entrenado"}

@router.post("/predecir")
def predecir(registro: RegistroML):
    prediccion = predecir_estado(registro.dict())
    return {"estado_animo_estimado": float(prediccion), "mensaje": "Predicción realizada"}

@router.get("/horarios")
def obtener_mejores_horarios():
    try:
        horarios = mejores_horarios("data/datos.csv")
        return {"mejores_horarios": horarios}
    except Exception as e:
        return {"error": str(e)}

