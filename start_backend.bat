@echo off
echo ========================================
echo   Backend Python - BERT & NLTK
echo ========================================
echo.

cd backend

echo Verification de Python...
python --version
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

echo.
echo Installation des dependances...
pip install -r requirements.txt

echo.
echo Demarrage du serveur Flask...
echo.
echo Le serveur sera disponible sur: http://localhost:5000
echo Appuyez sur Ctrl+C pour arreter le serveur
echo.

python app.py

pause 