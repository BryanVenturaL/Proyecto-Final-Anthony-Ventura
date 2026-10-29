@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python fases\03_reentrenamiento.py
pause

