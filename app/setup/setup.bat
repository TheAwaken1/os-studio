@echo off
rem ============================================================
rem  OS Studio - one-time setup
rem  Creates .\env (Python 3.10) and installs requirements.txt.
rem  Needs: an NVIDIA GPU with 16 GB VRAM, and either "uv" or
rem  Python 3.10 on this PC. Downloads about 4 GB.
rem ============================================================
setlocal
cd /d "%~dp0.."

where uv >nul 2>&1
if %errorlevel%==0 goto :with_uv
goto :with_pip

:with_uv
echo [1/3] Creating .\env with uv (Python 3.10)...
if not exist env\Scripts\python.exe uv venv --python 3.10 env || goto :fail
echo [2/3] Installing PyTorch (CUDA 13.0) and the app's packages...
uv pip install --python env\Scripts\python.exe -r requirements.txt --index-strategy unsafe-best-match || goto :fail
echo [3/3] Installing whisperx (word-accurate lyric timing)...
uv pip install --python env\Scripts\python.exe whisperx==3.8.6 --no-deps || goto :fail
goto :done

:with_pip
echo [1/3] Creating .\env with Python 3.10...
if not exist env\Scripts\python.exe py -3.10 -m venv env
if not exist env\Scripts\python.exe (
  echo.
  echo  Could not find Python 3.10. Install either:
  echo    uv           https://docs.astral.sh/uv/
  echo    Python 3.10  https://www.python.org/downloads/release/python-31011/
  goto :fail
)
env\Scripts\python.exe -m pip install -U pip >nul
echo [2/3] Installing PyTorch (CUDA 13.0) and the app's packages...
env\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
echo [3/3] Installing whisperx (word-accurate lyric timing)...
env\Scripts\python.exe -m pip install whisperx==3.8.6 --no-deps || goto :fail

:done
echo.
env\Scripts\python.exe -c "import torch; print('  torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
echo.
echo  Setup complete.
echo  Models: config.json -^> "models_dir" must point at a folder with YuE2\YuE2-3B
echo  and YuE2\YuE2-Vae (see README.md). Then run start.bat
echo.
pause
goto :eof

:fail
echo.
echo  SETUP FAILED - read the lines above.
pause
exit /b 1
