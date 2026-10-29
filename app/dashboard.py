import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st

from src import config
from src.predict import InterruptionPredictor

st.set_page_config(page_title="Interrupciones eléctricas", page_icon="⚡", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(config.PROCESSED_DATA, parse_dates=["inicio", "fin"])
    metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    reg = pd.read_csv(config.REPORTS_DIR / "metricas_duracion.csv")
    sev = pd.read_csv(config.REPORTS_DIR / "metricas_severidad.csv")
    return df, metadata, reg, sev


def options(df, column):
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values) if values else ["DESCONOCIDO"]


if not config.PROCESSED_DATA.exists() or not config.DURATION_MODEL.exists():
    st.error("Ejecute primero run_fase1.bat y run_fase2.bat")
    st.stop()

df, metadata, reg_metrics, sev_metrics = load_data()
st.title("⚡ Predicción de interrupciones eléctricas")
st.caption("Datos públicos de Osinergmin | Prototipo académico para apoyo operativo")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Interrupciones", f"{len(df):,}")
c2.metric("Duración mediana", f"{df['duracion_min'].median():.0f} min")
c3.metric("Modelo", metadata["modelo_duracion"])
c4.metric("MAE de prueba", f"{metadata['metricas_duracion']['MAE']:.1f} min")

tab1, tab2, tab3, tab4 = st.tabs(["Predicción", "Datos", "Modelos", "Metodología"])

with tab1:
    st.subheader("Nueva estimación")
    a, b, c = st.columns(3)
    with a:
        tipo = st.selectbox("Tipo", options(df, "tipo_interrupcion"))
        causa = st.selectbox("Causa", options(df, "causa_interrupcion"))
        cliente = st.selectbox("Cliente", options(df, "cliente"))
    with b:
        suministrador = st.selectbox("Suministrador", options(df, "suministrador"))
        propietario = st.selectbox("Propietario", options(df, "propietario"))
        nivel = st.selectbox("Nivel de tensión", options(df, "nivel_tension"))
    with c:
        tension = st.number_input("Tensión nominal (kV)", min_value=0.1, value=22.9)
        fecha = st.date_input("Fecha de inicio")
        hora = st.time_input("Hora de inicio")
        cantidad = st.number_input("Cantidad de suministradores", min_value=0, value=1)
    if st.button("Estimar interrupción", type="primary"):
        result = InterruptionPredictor().predict({
            "tipo_interrupcion": tipo,
            "causa_interrupcion": causa,
            "cliente": cliente,
            "suministrador": suministrador,
            "propietario": propietario,
            "nivel_tension": nivel,
            "tension_nominal_kv": tension,
            "fecha_hora_inicio": datetime.combine(fecha, hora),
            "cantidad_suministradores": cantidad,
        })
        r1, r2 = st.columns(2)
        r1.metric("Duración estimada", f"{result['duracion_estimada_min']:.1f} min")
        r2.metric("Severidad estimada", result["severidad_estimada"])
        if result["probabilidades_severidad"]:
            st.bar_chart(pd.Series(result["probabilidades_severidad"], name="Probabilidad"))

with tab2:
    left, right = st.columns(2)
    left.image(str(config.FIGURES_DIR / "01_distribucion_duracion.png"), width="stretch")
    right.bar_chart(df["causa_interrupcion"].fillna("Sin dato").value_counts())
    st.dataframe(
        df[["inicio", "id_punto_entrega", "tipo_interrupcion", "causa_interrupcion", "duracion_min", "severidad_eda"]].tail(100),
        width="stretch",
    )
    map_df = df[["latitud", "longitud"]].dropna().rename(columns={"latitud": "lat", "longitud": "lon"})
    if not map_df.empty and map_df["lat"].between(-20, 1).all() and map_df["lon"].between(-85, -65).all():
        st.map(map_df)

with tab3:
    left, right = st.columns(2)
    left.subheader("Duración")
    left.dataframe(reg_metrics, width="stretch")
    right.subheader("Severidad")
    right.dataframe(sev_metrics, width="stretch")
    p1, p2 = st.columns(2)
    p1.image(str(config.FIGURES_DIR / "02_comparacion_modelos.png"), width="stretch")
    p2.image(str(config.FIGURES_DIR / "03_real_vs_predicho.png"), width="stretch")
    p3, p4 = st.columns(2)
    p3.image(str(config.FIGURES_DIR / "04_matriz_confusion.png"), width="stretch")
    p4.image(str(config.FIGURES_DIR / "05_importancia_variables.png"), width="stretch")

with tab4:
    st.markdown(
        """
        **Flujo:** Osinergmin → limpieza → variables temporales y técnicas → división temporal 80/20 → comparación de modelos → API y dashboard.

        La severidad es un indicador académico basado en percentiles de duración: baja hasta el percentil 50, media hasta el percentil 80 y alta por encima de ese valor. No representa una categoría regulatoria.

        El modelo sirve para análisis y demostración. Una implementación operativa debe incorporar demanda afectada, usuarios, topología, protecciones y validación de especialistas.
        """
    )

