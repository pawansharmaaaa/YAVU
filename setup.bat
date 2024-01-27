REM Author: @pawansharmaaaa
REM Description: Setup the environment for the project

REM Create a virtual environment
python -m venv .venv

REM Activate the virtual environment
.venv\Scripts\activate

REM Install Pytorch for GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
pip install -r requirements.txt