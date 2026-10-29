@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python fases\02_modelos_y_evaluacion.py
pause

