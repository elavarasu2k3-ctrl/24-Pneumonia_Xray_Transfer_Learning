@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
call conda activate pneumonia_tf
python regen_gradcam.py > gradcam_log.txt 2>&1
echo DONE >> gradcam_log.txt
