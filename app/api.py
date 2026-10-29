import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.predict import InterruptionPredictor

app = FastAPI(
    title="API de duración y severidad de interrupciones",
    version="1.0.0",
    description="Modelo académico entrenado con datos públicos de Osinergmin.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InterruptionInput(BaseModel):
    tipo_interrupcion: str = "N"
    causa_interrupcion: str = "T"
    cliente: str = "EUC"
    suministrador: str = "ELP"
    propietario: str = "EUC"
    nivel_tension: str = "MT"
    tension_nominal_kv: float | None = Field(default=22.9, gt=0)
    fecha_hora_inicio: datetime
    cantidad_suministradores: int = Field(default=1, ge=0)


@app.get("/")
def root():
    return {"proyecto": "Predicción de interrupciones eléctricas", "documentacion": "/docs"}


@app.get("/health")
def health():
    ready = config.DURATION_MODEL.exists() and config.SEVERITY_MODEL.exists()
    return {"status": "ok" if ready else "sin_modelo", "modelo_disponible": ready}


@app.get("/metrics")
def metrics():
    if not config.METADATA_FILE.exists():
        raise HTTPException(404, "Modelo no entrenado")
    return json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: InterruptionInput):
    try:
        return InterruptionPredictor().predict(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Ejecute primero la fase 2") from exc


@app.get("/events")
def events(limit: int = 20):
    if not config.PROCESSED_DATA.exists():
        raise HTTPException(404, "Datos no procesados")
    df = pd.read_csv(config.PROCESSED_DATA).tail(min(max(limit, 1), 200))
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.post("/retrain")
def retrain():
    from src.train import train_models
    return train_models()

