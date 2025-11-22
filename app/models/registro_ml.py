from pydantic import BaseModel
from typing import List

# DTO que recibes desde Android (Room → Retrofit → Backend)
class SesionDTO(BaseModel):
    usuarioId: int
    rutinaId: int
    duracionSegundos: int
    estadoAnimo: int
    horaDelDia: str
    fecha: str

class HorarioItem(BaseModel):
    hora: float
    estadoAnimo: float

class HorarioResponse(BaseModel):
    mejores_horarios: List[HorarioItem]
    error: str | None = None


