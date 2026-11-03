@echo off
REM One-click runner: downloads the Kaggle dataset (if missing) and executes the notebook.
cd /d "%~dp0"
set LOG=run_log.txt
echo ==== Run started %date% %time% ==== > %LOG%

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3" >> %LOG% 2>&1
python --version >> %LOG% 2>&1

echo [1/4] Checking packages... >> %LOG%
python -c "import tensorflow" 2>nul || pip install "tensorflow>=2.15,<2.18" >> %LOG% 2>&1
pip install -q kaggle opencv-python-headless seaborn scikit-learn nbconvert ipykernel >> %LOG% 2>&1
python -c "import tensorflow as tf, sklearn; print('TF', tf.__version__, '| sklearn', sklearn.__version__)" >> %LOG% 2>&1

echo [2/4] Dataset... >> %LOG%
if not exist "data\chest_xray\train" (
    if not exist "data" mkdir data
    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data >> %LOG% 2>&1
    tar -xf "data\chest-xray-pneumonia.zip" -C data >> %LOG% 2>&1
)
if not exist "data\chest_xray\train" (
    echo ERROR: dataset not found after download - check kaggle.json >> %LOG%
    goto :end
)
echo Dataset ready. >> %LOG%

echo [3/4] Executing notebook (this takes 1-2 hours on CPU)... >> %LOG%
jupyter nbconvert --to notebook --execute Pneumonia_Transfer_Learning.ipynb --output Pneumonia_Transfer_Learning_executed.ipynb --ExecutePreprocessor.timeout=-1 >> %LOG% 2>&1
if errorlevel 1 (echo ERROR: notebook execution failed - see messages above >> %LOG%) else (echo [4/4] DONE - see Pneumonia_Transfer_Learning_executed.ipynb and artifacts\ >> %LOG%)

:end
echo ==== Run finished %date% %time% ==== >> %LOG%
type %LOG%
pause
