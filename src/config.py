from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

API_ROOT = (
    "https://gisem.osinergmin.gob.pe/serverosih/rest/services/"
    "Electricidad/MapaSEIN_Calidad/MapServer"
)
INTERRUPTIONS_LAYER = 2
DELIVERY_POINTS_LAYER = 0

RAW_INTERRUPTIONS = RAW_DIR / "interrupciones_osinergmin.json"
RAW_POINTS = RAW_DIR / "puntos_entrega_osinergmin.json"
PROCESSED_DATA = PROCESSED_DIR / "interrupciones_procesadas.csv"
DURATION_MODEL = MODELS_DIR / "modelo_duracion.pkl"
SEVERITY_MODEL = MODELS_DIR / "modelo_severidad.pkl"
METADATA_FILE = MODELS_DIR / "metadata.json"

for folder in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

