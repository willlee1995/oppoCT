import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

try:
    from totalsegmentator.python_api import totalsegmentator
    print("\nTotalSegmentator imported successfully.")
except ImportError as e:
    print(f"\nError importing TotalSegmentator: {e}")

try:
    import nnunetv2
    print(f"nnUNetv2 version: {nnunetv2.__version__}")
except ImportError:
    print("nnUNetv2 not found.")
