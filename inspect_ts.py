
from totalsegmentator.python_api import totalsegmentator
import inspect

print("=== TotalSegmentator API Signature ===")
sig = inspect.signature(totalsegmentator)
for param in sig.parameters.values():
    print(param)

print("\n=== Docstring ===")
print(totalsegmentator.__doc__)
