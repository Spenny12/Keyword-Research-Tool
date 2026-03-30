@echo off
echo Starting SEO Classifier Setup...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python from python.org
    pause
    exit /b
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating environment and installing requirements...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

echo.
echo ---------------------------------------------------
echo SEO Classifier is starting!
echo This will open a new tab in your browser.
echo Keep this window open while using the tool.
echo ---------------------------------------------------
echo.

streamlit run app.py
pause
