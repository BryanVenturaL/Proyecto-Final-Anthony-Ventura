from datetime import datetime

from src.features import FEATURES, prediction_frame


def test_prediction_frame_has_expected_features():
    frame = prediction_frame({"fecha_hora_inicio": datetime(2026, 10, 11, 23, 30)})
    assert list(frame.columns) == FEATURES
    assert frame.iloc[0]["es_noche"] == 1
    assert frame.iloc[0]["mes"] == 10

