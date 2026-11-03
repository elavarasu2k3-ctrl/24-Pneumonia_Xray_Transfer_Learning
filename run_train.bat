@echo off
REM Creates a Python 3.11 conda env with TensorFlow (Python 3.14 is not supported by TF) and executes the notebook.
cd /d "%~dp0"
set LOG=train_log.txt
echo ==== Train run started %date% %time% ==== > %LOG%
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3" >> %LOG% 2>&1

call conda env list | findstr /C:"pneumonia_tf" >nul
if errorlevel 1 (
    echo [1/4] Creating conda env pneumonia_tf with Python 3.11... >> %LOG%
    call conda create -y -n pneumonia_tf python=3.11 >> %LOG% 2>&1
)
call conda activate pneumonia_tf >> %LOG% 2>&1
python --version >> %LOG% 2>&1

echo [2/4] Installing packages... >> %LOG%
python -m pip install -q "tensorflow==2.17.1" numpy pandas scikit-learn matplotlib seaborn opencv-python-headless jupyter nbconvert ipykernel >> %LOG% 2>&1
python -m ipykernel install --user --name pneumonia_tf --display-name "Python (pneumonia_tf)" >> %LOG% 2>&1
python -c "import tensorflow as tf, sklearn, cv2; print('TF', tf.__version__, '| sklearn', sklearn.__version__)" >> %LOG% 2>&1
if errorlevel 1 (echo ERROR: TensorFlow import failed >> %LOG% & goto :end)

if not exist "data\chest_xray\train" (
    if exist "data\chest-xray-pneumonia.zip" tar -xf "data\chest-xray-pneumonia.zip" -C data >> %LOG% 2>&1
)
if not exist "data\chest_xray\train" (echo ERROR: dataset missing - data\chest_xray\train not found >> %LOG% & goto :end)
echo [3/4] Dataset ready. Executing notebook (1-2 hours on CPU)... >> %LOG%

jupyter nbconvert --to notebook --execute Pneumonia_Transfer_Learning.ipynb --output Pneumonia_Transfer_Learning_executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=pneumonia_tf >> %LOG% 2>&1
if errorlevel 1 (echo ERROR: notebook execution failed - see messages above >> %LOG%) else (echo [4/4] DONE - results in Pneumonia_Transfer_Learning_executed.ipynb and artifacts\ >> %LOG%)

:end
echo ==== Train run finished %date% %time% ==== >> %LOG%
