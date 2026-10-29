# Predicción de duración y severidad de interrupciones eléctricas

Proyecto integrador de Machine Learning aplicado a distribución y transmisión eléctrica. Utiliza información pública de Osinergmin para estimar la duración de una interrupción y clasificar su severidad operativa.

## Objetivo

Construir un pipeline reproducible que descargue interrupciones y puntos de entrega, valide los datos, compare modelos, publique predicciones mediante API y muestre los resultados en un dashboard.

La severidad es un indicador académico basado en la duración histórica:

- Baja: hasta el percentil 50.
- Media: entre los percentiles 50 y 80.
- Alta: por encima del percentil 80.

No corresponde a una categoría regulatoria. Una versión operativa debe incorporar usuarios, demanda afectada, topología, protecciones y energía no suministrada.

## Fuente pública

- Osinergmin, Mapa SEIN y calidad: `https://gisem.osinergmin.gob.pe/serverosih/rest/services/Electricidad/MapaSEIN_Calidad/MapServer`
- Capa 2: interrupciones.
- Capa 0: puntos de entrega, subestaciones, coordenadas y tensión.

La consulta realizada para esta versión devuelve 492 interrupciones de 2019 y 155 puntos de entrega.

## Arquitectura

```text
Osinergmin ArcGIS REST
        ↓
data/raw → limpieza → data/processed
        ↓
modelos de duración y severidad
        ↓
FastAPI + Streamlit + Docker
```

## Estructura

```text
app/            API y dashboard
data/           datos crudos y procesados
fases/          ejecución incremental por fase
models/         modelos y metadatos
reports/        métricas, predicciones y figuras
entregables/    informe, PDF y presentación final
src/            ingesta, limpieza, variables y modelos
tests/          prueba básica de variables
```

## Entregables finales

- `entregables/Informe_Proyecto_Final_Interrupciones.docx`: informe editable.
- `entregables/Informe_Proyecto_Final_Interrupciones.pdf`: informe listo para enviar.
- `entregables/Presentacion_Proyecto_Final_Interrupciones.pptx`: presentación de 12 diapositivas.
- `MANUAL_EXPOSICION.txt`: guion breve, demostración y respuestas probables.

## Instalación en Windows

1. Instalar Python 3.11 o superior.
2. Ejecutar `setup.bat`.
3. Ejecutar `run_fase1.bat` para descargar y procesar los datos.
4. Ejecutar `run_fase2.bat` para entrenar y evaluar los modelos.
5. Ejecutar `run_dashboard.bat` para abrir la plataforma.
6. Ejecutar `run_api.bat` para abrir la API.

Si el proyecto ya está instalado, basta con ejecutar `run_dashboard.bat`. Para la exposición, abrir también `run_api.bat` si se desea mostrar Swagger.

Direcciones locales:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Ejecución manual

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python fases\01_datos_y_eda.py
python fases\02_modelos_y_evaluacion.py
streamlit run app\dashboard.py
```

## Modelos

Duración:

- Baseline de mediana.
- Ridge.
- Ridge con transformación logarítmica del objetivo.
- Random Forest Regressor.
- Random Forest con transformación logarítmica del objetivo.

Severidad:

- Baseline de clase mayoritaria.
- Random Forest Classifier.

La selección usa un corte temporal 80/20. Para duración se minimiza MAE; para severidad se maximiza F1 macro.

## Resultado de la ejecución incluida

- Registros: 492; entrenamiento: 393; prueba temporal: 99.
- Duración: Ridge, MAE 115.0 minutos y R² 0.262.
- Severidad: Random Forest, F1 macro 0.551 y exactitud 72.7 %.
- Umbrales aprendidos: baja hasta 52 minutos y media hasta 373.7 minutos.

Estos resultados describen un prototipo con una muestra pública pequeña. Las métricas quedan guardadas en `reports/` para su revisión.

## API

`POST /predict`

```json
{
  "tipo_interrupcion": "N",
  "causa_interrupcion": "T",
  "cliente": "EUC",
  "suministrador": "ELP",
  "propietario": "EUC",
  "nivel_tension": "MT",
  "tension_nominal_kv": 22.9,
  "fecha_hora_inicio": "2026-10-11T19:30:00",
  "cantidad_suministradores": 1
}
```

También están disponibles `GET /health`, `GET /metrics`, `GET /events` y `POST /retrain`.

## Docker

```powershell
docker compose up --build
```

## Reentrenamiento

`run_fase3.bat` vuelve a consultar la fuente, reconstruye los datos, entrena los modelos y registra el cambio de versión y MAE en `reports/historial_reentrenamiento.json`.

## Limitaciones

- La muestra pública es pequeña y se usa como prototipo académico.
- Algunos campos contienen códigos cuya descripción debe solicitarse o validarse con documentación del regulador.
- La duración real solo se usa como objetivo y nunca como variable de entrada.
- El sistema no debe utilizarse para decisiones operativas sin validación técnica adicional.

