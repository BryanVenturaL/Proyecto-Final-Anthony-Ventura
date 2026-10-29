import json
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def _attributes(path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([x.get("attributes", {}) for x in payload.get("features", [])])


def _points() -> pd.DataFrame:
    payload = json.loads(config.RAW_POINTS.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("features", []):
        row = dict(item.get("attributes", {}))
        geometry = item.get("geometry") or {}
        row["LONGITUD"] = geometry.get("x", row.get("X"))
        row["LATITUD"] = geometry.get("y", row.get("Y"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    events = _attributes(config.RAW_INTERRUPTIONS)
    points = _points()
    events = events.drop_duplicates(subset="ID_PE_INT").copy()
    points = points.drop_duplicates(subset="ID_PE").copy()
    df = events.merge(points, on="ID_PE", how="left", suffixes=("", "_PUNTO"))

    df["inicio"] = pd.to_datetime(df["INT_INICIO"], unit="ms", utc=True, errors="coerce")
    df["fin"] = pd.to_datetime(df["INT_FIN"], unit="ms", utc=True, errors="coerce")
    df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    df = df[df["duracion_min"].between(1, 10080)].copy()

    df["hora"] = df["inicio"].dt.hour
    df["dia_semana"] = df["inicio"].dt.dayofweek
    df["mes"] = df["inicio"].dt.month
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_noche"] = ((df["hora"] < 6) | (df["hora"] >= 22)).astype(int)
    df["cantidad_suministradores"] = (
        df["SUMINISTRADOR"].fillna("").astype(str).str.split(",").map(lambda x: len([v for v in x if v.strip()]))
    )

    q50, q80 = df["duracion_min"].quantile([0.50, 0.80])
    df["severidad_eda"] = np.select(
        [df["duracion_min"] <= q50, df["duracion_min"] <= q80],
        ["Baja", "Media"],
        default="Alta",
    )

    rename = {
        "ID_PE_INT": "id_interrupcion",
        "ID_PE": "id_punto_entrega",
        "TIPO_INT": "tipo_interrupcion",
        "CAUSA_INT": "causa_interrupcion",
        "CLIENTE": "cliente",
        "SUMINISTRADOR": "suministrador",
        "OBSERVACION": "observacion",
        "NOM_SUB": "subestacion",
        "NOMBRE_PE": "punto_entrega",
        "PROPIETARIO": "propietario",
        "NIVEL_TENSION": "nivel_tension",
        "TENSION_NOM_OP": "tension_nominal_kv",
        "LONGITUD": "longitud",
        "LATITUD": "latitud",
    }
    df = df.rename(columns=rename)
    keep = [
        "id_interrupcion", "id_punto_entrega", "inicio", "fin", "duracion_min",
        "tipo_interrupcion", "causa_interrupcion", "cliente", "suministrador",
        "propietario", "subestacion", "punto_entrega", "nivel_tension",
        "tension_nominal_kv", "longitud", "latitud", "observacion", "hora",
        "dia_semana", "mes", "es_fin_semana", "es_noche",
        "cantidad_suministradores", "severidad_eda",
    ]
    for column in keep:
        if column not in df:
            df[column] = np.nan
    df = df[keep].sort_values("inicio").reset_index(drop=True)
    df.to_csv(config.PROCESSED_DATA, index=False, encoding="utf-8-sig")
    log.info("Datos procesados: %s registros", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(build_dataset().shape)
