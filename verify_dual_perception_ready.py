#!/usr/bin/env python3
"""
Quick verification script to ensure dual perception is ready for training
"""

import sys
import os

print("🔍 Verifying Dual Perception Setup...")
print("=" * 60)

# 1. Check if dual_perception.py exists
print("\n1️⃣ Checking dual_perception module...")
if os.path.exists("listener/dual_perception.py"):
    print("   ✅ listener/dual_perception.py exists")
else:
    print("   ❌ listener/dual_perception.py NOT FOUND!")
    sys.exit(1)

# 2. Try importing DualPerceptionModule
print("\n2️⃣ Testing DualPerceptionModule import...")
try:
    from listener.dual_perception import DualPerceptionModule
    print("   ✅ DualPerceptionModule import successful")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# 3. Check if Chandra_trainer has the new code
print("\n3️⃣ Checking Chandra_trainer.py for dual perception code...")
with open("Chandra_trainer.py", "r") as f:
    content = f.read()
    if "🎵 Dual perception enabled:" in content:
        print("   ✅ New dual perception initialization code found")
    else:
        print("   ❌ Dual perception code NOT FOUND in Chandra_trainer.py")
        sys.exit(1)
    
    if "_augment_with_dual_features" in content:
        print("   ✅ _augment_with_dual_features method found")
    else:
        print("   ❌ _augment_with_dual_features method NOT FOUND")
        sys.exit(1)

# 4. Check MPS/GPU availability
print("\n4️⃣ Checking GPU availability...")
try:
    import torch
    if torch.backends.mps.is_available():
        print("   ✅ MPS (Apple Silicon GPU) available")
    elif torch.cuda.is_available():
        print("   ✅ CUDA GPU available")
    else:
        print("   ⚠️  No GPU detected - training will use CPU (slower)")
except ImportError:
    print("   ⚠️  PyTorch not found - can't check GPU")

# 5. Check if Georgia.wav exists
print("\n5️⃣ Checking input audio...")
if os.path.exists("input_audio/Georgia.wav"):
    print("   ✅ input_audio/Georgia.wav exists")
    # Get file size
    size_mb = os.path.getsize("input_audio/Georgia.wav") / (1024 * 1024)
    print(f"   📊 File size: {size_mb:.2f} MB")
else:
    print("   ❌ input_audio/Georgia.wav NOT FOUND")
    sys.exit(1)

# 6. Check if JSON directory exists
print("\n6️⃣ Checking output directory...")
if os.path.exists("JSON"):
    print("   ✅ JSON/ directory exists")
else:
    print("   ⚠️  JSON/ directory doesn't exist, creating it...")
    os.makedirs("JSON")
    print("   ✅ JSON/ directory created")

# 7. Check required dependencies
print("\n7️⃣ Checking dependencies...")
required = ['librosa', 'numpy', 'torch', 'transformers']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} NOT FOUND")
        missing.append(pkg)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("   Install with: pip install " + " ".join(missing))
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("\nYou're ready to retrain Georgia with dual perception:")
print("=" * 60)
print("python Chandra_trainer.py \\")
print("    --file input_audio/Georgia.wav \\")
print("    --hybrid-perception \\")
print("    --wav2vec \\")
print("    --gpu")
print("=" * 60)
print("\nExpected output should include:")
print("  🎵 Dual perception enabled:")
print("     Machine logic: facebook/wav2vec2-base → gesture tokens (0-63)")
print("     Machine logic: Ratio analysis → consonance + frequency ratios")
print("     ✨ Tokens ARE the patterns, not chord names!")
print("=" * 60)





