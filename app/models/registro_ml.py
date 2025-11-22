from pydantic import BaseModel

class RegistroML(BaseModel):
    duracion: int
    tecnica: int
    estadoAnimo: int
    hora: float
    dia: int

