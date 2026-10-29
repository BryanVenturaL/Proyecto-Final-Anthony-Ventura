import json
import logging
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features import FEATURES, model_frame
from src.simple_models import LogTargetRegressor, MajorityClassifier, MedianRegressor, ModelBundle, RandomForest, RidgeRegressor

log = logging.getLogger(__name__)


def _severity(values, q50, q80):
    return np.select([values <= q50, values <= q80], ["Baja", "Media"], default="Alta")


def _reg_metrics(y, pred):
    residual = np.asarray(y) - pred
    denominator = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": float(1 - np.sum(residual ** 2) / denominator) if denominator else 0.0,
    }


def _class_metrics(y, pred):
    y, pred, recalls, f1s = np.asarray(y), np.asarray(pred), [], []
    for cls in np.unique(y):
        tp = np.sum((y == cls) & (pred == cls))
        fp = np.sum((y != cls) & (pred == cls))
        fn = np.sum((y == cls) & (pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return {
        "F1_macro": float(np.mean(f1s)),
        "exactitud": float(np.mean(y == pred)),
        "exactitud_balanceada": float(np.mean(recalls)),
    }


def train_models():
    df = pd.read_csv(config.PROCESSED_DATA, parse_dates=["inicio", "fin"]).sort_values("inicio")
    split = int(len(df) * 0.80)
    train, test = df.iloc[:split], df.iloc[split:]
    x_train, x_test = model_frame(train), model_frame(test)
    y_train, y_test = train["duracion_min"].to_numpy(), test["duracion_min"].to_numpy()

    regressors = {
        "Baseline mediana": ModelBundle(MedianRegressor()),
        "Ridge": ModelBundle(RidgeRegressor(alpha=10.0)),
        "Ridge log": ModelBundle(LogTargetRegressor(RidgeRegressor(alpha=10.0))),
        "Random Forest": ModelBundle(RandomForest("regression", n_estimators=55)),
        "Random Forest log": ModelBundle(LogTargetRegressor(RandomForest("regression", n_estimators=55))),
    }
    reg_rows, reg_fitted = [], {}
    for name, model in regressors.items():
        model.fit(x_train, y_train)
        prediction = np.maximum(model.predict(x_test).astype(float), 1)
        reg_fitted[name] = (model, prediction)
        reg_rows.append({"modelo": name, **_reg_metrics(y_test, prediction)})
    reg_results = pd.DataFrame(reg_rows).sort_values("MAE")
    best_reg_name = reg_results.iloc[0]["modelo"]
    best_prediction = reg_fitted[best_reg_name][1]

    q50, q80 = np.quantile(y_train, [0.50, 0.80])
    sev_train, sev_test = _severity(y_train, q50, q80), _severity(y_test, q50, q80)
    classifiers = {
        "Baseline mayoría": ModelBundle(MajorityClassifier()),
        "Random Forest": ModelBundle(RandomForest("classification", n_estimators=75)),
    }
    cls_rows, cls_fitted = [], {}
    for name, model in classifiers.items():
        model.fit(x_train, sev_train)
        prediction = model.predict(x_test)
        cls_fitted[name] = (model, prediction)
        cls_rows.append({"modelo": name, **_class_metrics(sev_test, prediction)})
    cls_results = pd.DataFrame(cls_rows).sort_values("F1_macro", ascending=False)
    best_cls_name = cls_results.iloc[0]["modelo"]
    best_cls_prediction = cls_fitted[best_cls_name][1]

    production_reg = regressors[best_reg_name].fit(model_frame(df), df["duracion_min"].to_numpy())
    production_cls = classifiers[best_cls_name].fit(model_frame(df), _severity(df["duracion_min"].to_numpy(), q50, q80))
    config.DURATION_MODEL.write_bytes(pickle.dumps(production_reg))
    config.SEVERITY_MODEL.write_bytes(pickle.dumps(production_cls))

    reg_results.to_csv(config.REPORTS_DIR / "metricas_duracion.csv", index=False)
    cls_results.to_csv(config.REPORTS_DIR / "metricas_severidad.csv", index=False)
    pd.DataFrame({
        "inicio": test["inicio"], "real_min": y_test, "predicho_min": best_prediction,
        "severidad_real": sev_test, "severidad_predicha": best_cls_prediction,
    }).to_csv(config.REPORTS_DIR / "predicciones_prueba.csv", index=False)

    importance = pd.DataFrame(columns=["variable", "importancia"])
    if hasattr(production_reg.estimator, "feature_importances_"):
        importance = pd.DataFrame({
            "variable": production_reg.feature_names_,
            "importancia": production_reg.feature_importances_,
        }).sort_values("importancia", ascending=False)
    importance.to_csv(config.REPORTS_DIR / "importancia_variables.csv", index=False)

    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(),
        "fuente": config.API_ROOT,
        "registros": len(df), "entrenamiento": len(train), "prueba": len(test),
        "modelo_duracion": best_reg_name, "modelo_severidad": best_cls_name,
        "metricas_duracion": reg_results.iloc[0].to_dict(),
        "metricas_severidad": cls_results.iloc[0].to_dict(),
        "umbrales_severidad_min": {"baja_hasta": float(q50), "media_hasta": float(q80)},
        "variables": FEATURES,
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _plots(df, reg_results, y_test, best_prediction, sev_test, best_cls_prediction, importance)
    log.info("Modelos guardados: %s / %s", best_reg_name, best_cls_name)
    return metadata


def _plots(df, metrics, y_test, prediction, sev_test, sev_prediction, importance):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    df["duracion_min"].clip(upper=df["duracion_min"].quantile(0.95)).hist(ax=ax, bins=25, color="#155eef")
    ax.set(title="Distribución de duración de interrupciones", xlabel="Minutos", ylabel="Eventos")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "01_distribucion_duracion.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics.sort_values("MAE").plot.barh(x="modelo", y="MAE", ax=ax, legend=False, color="#f79009")
    ax.set(title="Comparación de modelos", xlabel="MAE (minutos)", ylabel="")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "02_comparacion_modelos.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, prediction, alpha=0.7, color="#155eef")
    limit = max(float(np.max(y_test)), float(np.max(prediction)))
    ax.plot([0, limit], [0, limit], "--", color="#d92d20")
    ax.set(title="Duración real frente a predicha", xlabel="Real (min)", ylabel="Predicha (min)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "03_real_vs_predicho.png", dpi=160); plt.close(fig)
    labels = ["Baja", "Media", "Alta"]
    matrix = np.array([[np.sum((sev_test == a) & (sev_prediction == b)) for b in labels] for a in labels])
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
           xlabel="Predicha", ylabel="Real", title="Clasificación de severidad")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "04_matriz_confusion.png", dpi=160); plt.close(fig)
    if not importance.empty:
        top = importance.head(12).sort_values("importancia")
        fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(top["variable"], top["importancia"], color="#12b76a")
        ax.set(title="Variables con mayor influencia", xlabel="Importancia")
        fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "05_importancia_variables.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(train_models())
