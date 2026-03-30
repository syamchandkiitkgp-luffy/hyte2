import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import get_embedding

print("Testing embedding...")
vec = get_embedding("test")
if vec:
    print(f"Success! Vector length: {len(vec)}")
else:
    print("Failed to get embedding.")
