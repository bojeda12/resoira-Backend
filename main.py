from fastapi import FastAPI
from app.routes import ml_routes

app = FastAPI(title="Backend ML Respiración")

# Registrar rutas
app.include_router(ml_routes.router)

