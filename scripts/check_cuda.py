#!/usr/bin/env python
"""
Helper script to check PyTorch CUDA availability and provide installation instructions.
"""

import sys
import platform
import importlib.metadata

def check_pytorch_cuda():
    """Check if PyTorch can detect CUDA."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    cuda_available = False
    
    try:
        import torch
        print("\n" + "=" * 60)
        print("PyTorch CUDA Status Check")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            cuda_available = True
        else:
            print("\nWARNING: PyTorch cannot detect a GPU.")
            print("Please ensure you have installed the CUDA-enabled version of PyTorch.")
            print("Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
    except ImportError:
        print("\nERROR: PyTorch is not installed.")
        return False

    try:
        print("\n" + "=" * 60)
        print("TotalSegmentator Check")
        print("=" * 60)
        try:
             ts_version = importlib.metadata.version("totalsegmentator")
             print(f"TotalSegmentator version: {ts_version}")
        except importlib.metadata.PackageNotFoundError:
             print("TotalSegmentator NOT installed.")
             
    except Exception as e:
        print(f"Error checking TotalSegmentator: {e}")
    
    return cuda_available

if __name__ == "__main__":
    check_pytorch_cuda()

