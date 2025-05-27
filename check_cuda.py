# Run this script to check if the environment can recognize and utilize the GPU resources for training
# If PyTorch CUDA is unavailable, download the dependencies here: https://pytorch.org/get-started/locally/
# To check the CUDA version of your GPU, run 'nvidia-smi' in your terminal. You should see the CUDA version listed on the top right.

'''
Example installation of PyTorch CUDA (for compute platform version 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

'''

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Current GPU ID: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available to PyTorch. This is the cause of the error.")