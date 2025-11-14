#!/usr/bin/env python3
"""
Convenience wrapper for train_modular.py
Allows running training from root directory with simple command:
    python train.py audio.wav output.json
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main training script
from scripts.train.train_modular import main

if __name__ == '__main__':
    sys.exit(main())
