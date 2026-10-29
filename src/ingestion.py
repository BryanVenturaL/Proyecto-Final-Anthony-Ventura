import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src import config

log = logging.getLogger(__name__)


def _request(layer: int, offset: int, page_size: int) -> dict:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "json",
    }
    url = f"{config.API_ROOT}/{layer}/query?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "ProyectoAcademico/1.0"})
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_layer(layer: int, page_size: int = 500) -> list[dict]:
    rows, offset = [], 0
    while True:
        payload = _request(layer, offset, page_size)
        batch = payload.get("features", [])
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if not payload.get("exceededTransferLimit", False):
            break
    return rows


def ingest_all() -> dict:
    interruptions = download_layer(config.INTERRUPTIONS_LAYER)
    points = download_layer(config.DELIVERY_POINTS_LAYER)
    stamp = datetime.now(timezone.utc).isoformat()
    config.RAW_INTERRUPTIONS.write_text(
        json.dumps({"downloaded_at": stamp, "features": interruptions}, ensure_ascii=False),
        encoding="utf-8",
    )
    config.RAW_POINTS.write_text(
        json.dumps({"downloaded_at": stamp, "features": points}, ensure_ascii=False),
        encoding="utf-8",
    )
    result = {"interrupciones": len(interruptions), "puntos_entrega": len(points)}
    log.info("Descarga completada: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(ingest_all())

