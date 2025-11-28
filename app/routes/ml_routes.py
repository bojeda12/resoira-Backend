from fastapi import APIRouter
from typing import List
from app.models.registro_ml import SesionDTO, HorarioResponse
from app.services.red_neuronal import (
    predecir_estado,
    mejores_horarios,
    calcular_horarios,
    entrenar_completo,
    entrenar_incremental,
    convertir_hora,
    convertir_dia
)

router = APIRouter()

@router.post("/predecir", response_model=HorarioResponse)
def predecir(sesiones: List[SesionDTO]):
    try:
        # Mientras el modelo esté en fase de entrenamiento completo (<100)
        if len(sesiones) < 100:
            # Caso inicial: muy pocos datos (<5)
            if len(sesiones) < 5:
                return HorarioResponse(
                    mejores_horarios=[],
                    error="Estamos aprendiendo de ti. Registra cómo te sientes y realiza al menos una sesión diaria."
                )

            # Caso intermedio: entre 5 y 99 → entrenar completo pero aún mostrar mensaje motivador
            entrenar_completo("data/datos.csv")
            horarios = calcular_horarios(sesiones)
            return HorarioResponse(
                mejores_horarios=horarios,
                error="Estamos aprendiendo de ti. El modelo sigue ajustándose a tus patrones"
            )

        # Una vez que hay suficientes datos (≥100) → entrenar incremental, sin mensaje
        else:
            X_new = [
                [s.duracionSegundos, s.rutinaId, s.horaDelDia, convertir_dia(s.fecha)]
                for s in sesiones
            ]


            y_new = [s.estadoAnimo for s in sesiones]
            entrenar_incremental(X_new, y_new)

            horarios = calcular_horarios(sesiones)
            return HorarioResponse(mejores_horarios=horarios)

    except Exception as e:
        return HorarioResponse(mejores_horarios=[], error=str(e))


@router.post("/estado")
def predecir_estado_animo(sesion: SesionDTO):
    """
    Recibe una sola sesión y devuelve la predicción de estado de ánimo
    usando el modelo entrenado.
    """
    try:
        resultado = predecir_estado(sesion.dict())
        return {"estadoAnimoPredicho": resultado}
    except Exception as e:
        return {"error": str(e)}


@router.get("/mejores")
def obtener_mejores_horarios():
    """
    Endpoint auxiliar que lee el CSV de datos y devuelve los mejores horarios
    calculados con promedios de estado de ánimo.
    """
    try:
        mejores = mejores_horarios("data/datos.csv", top_n=3)
        return {"mejores_horarios": mejores}
    except Exception as e:
        return {"error": str(e)}

