@echo off
setlocal

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat

    echo Installing requirements...
    echo pillow==10.3.0 > requirements.txt
    echo opencv-python==4.9.0.80 >> requirements.txt
    echo numpy==1.26.4 >> requirements.txt
    echo onnxruntime==1.17.1 >> requirements.txt
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)
python coinGui.py

endlocal