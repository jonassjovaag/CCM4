#!/usr/bin/env python3
"""
Convenience wrapper for MusicHal_9000.py
Allows running live performance from root directory with simple command:
    python perform.py [options]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run MusicHal_9000
script_path = project_root / 'scripts' / 'performance' / 'MusicHal_9000.py'

with open(script_path) as f:
    code = compile(f.read(), script_path, 'exec')
    exec(code)
