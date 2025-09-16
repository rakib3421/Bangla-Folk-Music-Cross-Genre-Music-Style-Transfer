@echo off
REM Deployment script for Windows
REM This script sets up the Style Transfer application on Windows

echo 🚀 Starting Style Transfer Application Deployment

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.11+ first.
    exit /b 1
)

REM Check if Redis is available (optional)
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Redis is not running. Install and start Redis for optimal performance.
    echo    You can download Redis from: https://github.com/microsoftarchive/redis/releases
)

REM Create virtual environment
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt
pip install -r requirements-web.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "app\static\uploads" mkdir "app\static\uploads"
if not exist "cache" mkdir "cache"
if not exist "logs" mkdir "logs"

REM Copy environment file if it doesn't exist
if not exist .env (
    echo 📝 Creating environment file...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your configuration
)

REM Check if we can import the app
echo 🧪 Testing application import...
python -c "from app import create_app; print('✅ App import successful')" || (
    echo ❌ Failed to import application. Check dependencies.
    exit /b 1
)

echo ✅ Setup complete!
echo.
echo 🎵 To start the application:
echo    python run_app.py
echo.
echo 🌐 The web interface will be available at: http://localhost:5000
echo 📡 API documentation at: http://localhost:5000/api/v1/info

pause