#!/usr/bin/env python
"""
Helper script to check PyTorch CUDA availability and provide installation instructions.
"""

import sys


def check_pytorch_cuda():
    """Check if PyTorch can detect CUDA."""
    try:
        import torch
        print("=" * 60)
        print("PyTorch CUDA Status Check")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("\n✓ PyTorch can detect your GPU!")
        else:
            print("\n✗ PyTorch cannot detect CUDA/GPU")
            print("\nTo fix this, reinstall PyTorch with CUDA support:")
            print("\nFor CUDA 11.8:")
            print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\nFor CUDA 12.1:")
            print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\nFor CUDA 12.4:")
            print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("\nTo check your CUDA version:")
            print("  nvcc --version")
            print("  nvidia-smi")
        
        print("=" * 60)
        return torch.cuda.is_available()
        
    except ImportError:
        print("PyTorch is not installed.")
        print("Install it with: uv pip install torch")
        return False

if __name__ == '__main__':
    success = check_pytorch_cuda()
    sys.exit(0 if success else 1)








