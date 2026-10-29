import json
import pickle

from src import config
from src.features import prediction_frame


class InterruptionPredictor:
    def __init__(self):
        self.duration_model = pickle.loads(config.DURATION_MODEL.read_bytes())
        self.severity_model = pickle.loads(config.SEVERITY_MODEL.read_bytes())
        self.metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))

    def predict(self, data: dict) -> dict:
        frame = prediction_frame(data)
        duration = max(float(self.duration_model.predict(frame)[0]), 1.0)
        severity = str(self.severity_model.predict(frame)[0])
        probabilities = {}
        if hasattr(self.severity_model, "predict_proba"):
            values = self.severity_model.predict_proba(frame)[0]
            classes = self.severity_model.classes_
            probabilities = {str(k): round(float(v), 4) for k, v in zip(classes, values)}
        return {
            "duracion_estimada_min": round(duration, 1),
            "duracion_estimada_horas": round(duration / 60, 2),
            "severidad_estimada": severity,
            "probabilidades_severidad": probabilities,
            "modelo_duracion": self.metadata["modelo_duracion"],
            "version": self.metadata["version"],
        }
