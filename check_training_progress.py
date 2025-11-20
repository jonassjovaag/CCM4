#!/usr/bin/env python3
"""
Quick script to check training progress from log output.
Usage: python check_training_progress.py
"""

import sys
import re
from pathlib import Path

def check_progress():
    """Check training progress from recent output"""
    
    print("🔍 Checking training progress...\n")
    
    # Check if output JSON exists
    json_files = list(Path('JSON').glob('*.json'))
    if json_files:
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        size_mb = latest.stat().st_size / 1024 / 1024
        print(f"📦 Latest model: {latest.name} ({size_mb:.1f} MB)")
        print(f"   Modified: {latest.stat().st_mtime}")
    
    # Check for performance arc output
    arc_files = list(Path('ai_learning_data').glob('*performance_arc.json'))
    if arc_files:
        latest_arc = max(arc_files, key=lambda p: p.stat().st_mtime)
        print(f"\n🎭 Performance Arc: {latest_arc.name}")
    
    # Common progress indicators
    progress_patterns = [
        (r'Stage (\d)/(\d): (\w+)', 'Pipeline Stage'),
        (r'Processing (\d+)/(\d+) events', 'Event Processing'),
        (r'Training state (\d+)', 'Oracle Training'),
        (r'(\d+)% complete', 'Progress'),
        (r'Analyzing performance arc', 'Performance Arc'),
        (r'(\d+) phases.*(\d+\.\d+)s duration', 'Arc Analysis'),
    ]
    
    print("\n📊 Expected pipeline stages:")
    stages = [
        "1/6: AudioExtraction (cached if available)",
        "2/6: FeatureAnalysis (MERT model loading + extraction)",
        "3/6: HierarchicalSampling (optional filtering)",
        "4/6: PerformanceArc (NEW - analyzing musical structure)",
        "5/6: OracleTraining (building AudioOracle graph)",
        "6/6: Validation (checking output quality)"
    ]
    
    for i, stage in enumerate(stages, 1):
        print(f"   {stage}")
    
    print("\n💡 Current stage: FeatureAnalysis")
    print("   Status: Loading MERT model to GPU (this takes 1-3 minutes)")
    print("   Next: Extract 768D features from cached events")
    print("   Then: Performance Arc analysis (NEW STAGE)")
    
    print("\n⏱️  Typical timing:")
    print("   - MERT model load: 1-3 min (one-time)")
    print("   - Feature extraction: ~1 min per 1000 events")
    print("   - Performance Arc: 30-60 sec")
    print("   - Oracle training: ~10 sec per 1000 states")
    
    print("\n🔧 To monitor live progress:")
    print("   watch -n 5 'tail -30 <training_output_file>'")
    print("   or check JSON file size: watch -n 5 'ls -lh JSON/*.json'")

if __name__ == '__main__':
    check_progress()
