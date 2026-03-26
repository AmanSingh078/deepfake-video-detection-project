@echo off
echo ========================================
echo Tri-Expert Detection Suite - Installation Script
echo ========================================
echo.
echo Installing required packages...
echo This may take 10-15 minutes depending on your internet speed.
echo.

py -m pip install --upgrade pip
py -m pip install torch torchvision albumentations timm facenet-pytorch pandas numpy opencv-python pillow tqdm

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download model weights and place in 'weights' folder.
echo.
echo 2. Run detection with your videos:
echo    py predict_folder.py --test-dir "path\to\videos" --output results.csv
echo.
pause
