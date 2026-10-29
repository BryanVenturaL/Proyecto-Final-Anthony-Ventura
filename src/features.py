from datetime import datetime

import pandas as pd

CATEGORICAL = [
    "tipo_interrupcion",
    "causa_interrupcion",
    "cliente",
    "suministrador",
    "propietario",
    "nivel_tension",
]
NUMERIC = [
    "tension_nominal_kv",
    "hora",
    "dia_semana",
    "mes",
    "es_fin_semana",
    "es_noche",
    "cantidad_suministradores",
]
FEATURES = CATEGORICAL + NUMERIC


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in CATEGORICAL:
        output[column] = output[column].fillna("DESCONOCIDO").astype(str)
    for column in NUMERIC:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[FEATURES]


def prediction_frame(data: dict) -> pd.DataFrame:
    start = data.get("fecha_hora_inicio")
    start = pd.Timestamp(start if start else datetime.now())
    row = {
        "tipo_interrupcion": data.get("tipo_interrupcion", "N"),
        "causa_interrupcion": data.get("causa_interrupcion", "T"),
        "cliente": data.get("cliente", "DESCONOCIDO"),
        "suministrador": data.get("suministrador", "DESCONOCIDO"),
        "propietario": data.get("propietario", "DESCONOCIDO"),
        "nivel_tension": data.get("nivel_tension", "DESCONOCIDO"),
        "tension_nominal_kv": data.get("tension_nominal_kv"),
        "hora": start.hour,
        "dia_semana": start.dayofweek,
        "mes": start.month,
        "es_fin_semana": int(start.dayofweek >= 5),
        "es_noche": int(start.hour < 6 or start.hour >= 22),
        "cantidad_suministradores": data.get("cantidad_suministradores", 1),
    }
    return model_frame(pd.DataFrame([row]))

