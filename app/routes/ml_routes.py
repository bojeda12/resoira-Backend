from fastapi import APIRouter
from typing import List
from app.models.registro_ml import SesionDTO, HorarioResponse
from app.services.red_neuronal import predecir_estado, mejores_horarios,calcular_horarios


router = APIRouter()

@router.post("/predecir", response_model=HorarioResponse)
def predecir(sesiones: List[SesionDTO]):
    """
    Recibe una lista de sesiones enviadas desde Room (Android),
    calcula los mejores horarios y devuelve la respuesta.
    """
    try:
        # Si aún no hay suficientes datos, devolvemos un mensaje motivador
        if len(sesiones) < 5:
            return HorarioResponse(
                mejores_horarios=[],
                error="Estamos aprendiendo de ti. Registra cómo te sientes y realiza al menos una sesión diaria."
            )

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


