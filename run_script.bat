@echo off
pushd "%~dp0"
call venv\Scripts\activate.bat
python your_script.py
popd
