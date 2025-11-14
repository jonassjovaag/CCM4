#!/usr/bin/env python3
"""
Test script to isolate the hanging issue in MusicHal_9000.py
"""

# FIRST: Ensure CCM3 virtual environment is active
try:
    from ccm3_venv_manager import ensure_ccm3_venv_active
    ensure_ccm3_venv_active()
    print("✅ CCM3 virtual environment activated")
except ImportError:
    print("⚠️  CCM3 virtual environment manager not found, continuing with current environment")
except Exception as e:
    print(f"⚠️  CCM3 virtual environment activation failed: {e}")
    print("Continuing with current environment...")

print("Testing imports step by step...")

# Basic imports
import time
import threading
import argparse
print("✅ Basic imports OK")

# Start testing MusicHal imports
try:
    from listener.jhs_listener_core import DriftListener, Event
    print("✅ listener.jhs_listener_core OK")
except Exception as e:
    print(f"❌ listener.jhs_listener_core FAILED: {e}")

try:
    from rhythmic_engine.audio_file_learning.lightweight_rhythmic_analyzer import LightweightRhythmicAnalyzer
    print("✅ rhythmic_engine.lightweight_rhythmic_analyzer OK")
except Exception as e:
    print(f"❌ rhythmic_engine.lightweight_rhythmic_analyzer FAILED: {e}")

try:
    from correlation_engine.unified_decision_engine import UnifiedDecisionEngine, CrossModalContext, MusicalContext
    print("✅ correlation_engine.unified_decision_engine OK")
except Exception as e:
    print(f"❌ correlation_engine.unified_decision_engine FAILED: {e}")

try:
    from listener.hybrid_detector import HybridDetector, DetectionResult
    print("✅ listener.hybrid_detector OK")
except Exception as e:
    print(f"❌ listener.hybrid_detector FAILED: {e}")

print("✅ All imports tested - now testing argparse...")

# Test argparse
parser = argparse.ArgumentParser(description='Test Enhanced Drift Engine AI')
parser.add_argument('--test', action='store_true', help='Test argument')
parser.add_argument('--performance-duration', type=int, default=0, help='Performance duration in minutes (0 = no timeline)')
parser.add_argument('--hybrid-perception', action='store_true', help='Enable hybrid perception')

print("✅ Argparse setup complete - parsing arguments...")

args = parser.parse_args()
print(f"✅ Arguments parsed successfully: {args}")