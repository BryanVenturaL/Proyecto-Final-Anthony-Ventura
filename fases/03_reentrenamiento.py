import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.ingestion import ingest_all
from src.preprocessing import build_dataset
from src.train import train_models


def main():
    old = json.loads(config.METADATA_FILE.read_text(encoding="utf-8")) if config.METADATA_FILE.exists() else {}
    counts = ingest_all()
    df = build_dataset()
    new = train_models()
    log_file = config.REPORTS_DIR / "historial_reentrenamiento.json"
    history = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else []
    history.append({
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "registros_descargados": counts,
        "registros_validos": len(df),
        "version_anterior": old.get("version"),
        "version_nueva": new["version"],
        "mae_anterior": old.get("metricas_duracion", {}).get("MAE"),
        "mae_nuevo": new["metricas_duracion"]["MAE"],
    })
    log_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(history[-1])


if __name__ == "__main__":
    main()
