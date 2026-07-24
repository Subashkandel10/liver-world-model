"""Put the repo root on sys.path so `import lwm...` resolves when pytest runs from a fresh checkout."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
