import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt

from src import config
from src.ingestion import ingest_all
from src.preprocessing import build_dataset


def main():
    counts = ingest_all()
    df = build_dataset()
    summary = df[["duracion_min", "tension_nominal_kv", "cantidad_suministradores"]].describe().T
    summary.to_csv(config.REPORTS_DIR / "estadistica_descriptiva.csv")
    by_cause = df.groupby("causa_interrupcion", dropna=False)["duracion_min"].agg(["count", "median", "mean"])
    by_cause.to_csv(config.REPORTS_DIR / "duracion_por_causa.csv")
    print({**counts, "registros_validos": len(df)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()

