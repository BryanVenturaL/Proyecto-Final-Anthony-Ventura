@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python fases\01_datos_y_eda.py
pause

