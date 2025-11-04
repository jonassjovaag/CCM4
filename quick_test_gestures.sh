#!/bin/bash
# Quick test to compare token diversity - stops after token assignment

echo "🧪 Quick Gesture Token Diversity Test"
echo "====================================="
echo ""
echo "This will take ~5-7 minutes per test (stops after token assignment)"
echo ""

# Test with peak consolidation (current approach)
echo "=== Testing Peak Consolidation ==="
timeout 600 python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_peak.json" \
    --max-events 2000 \
    --training-events 100 \
    --no-temporal-smoothing \
    --use-musical-gestures \
    --gesture-consolidation peak \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | tee /tmp/test_peak.log

echo ""
echo "Peak consolidation result:"
grep "Unique tokens assigned:" /tmp/test_peak.log || echo "❌ No token data found"
echo ""
echo "---"
echo ""

# Test with first consolidation
echo "=== Testing First Consolidation ==="
timeout 600 python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_first.json" \
    --max-events 2000 \
    --training-events 100 \
    --no-temporal-smoothing \
    --use-musical-gestures \
    --gesture-consolidation first \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | tee /tmp/test_first.log

echo ""
echo "First consolidation result:"
grep "Unique tokens assigned:" /tmp/test_first.log || echo "❌ No token data found"
echo ""
echo "---"
echo ""

# Test without gestures (raw)
echo "=== Testing Raw (No Gestures) ==="
timeout 600 python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_raw.json" \
    --max-events 2000 \
    --training-events 100 \
    --no-temporal-smoothing \
    --no-musical-gestures \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | tee /tmp/test_raw.log

echo ""
echo "Raw (no gestures) result:"
grep "Unique tokens assigned:" /tmp/test_raw.log || echo "❌ No token data found"
echo ""

echo "=== SUMMARY ==="
echo "Peak consolidation:"
grep "Unique tokens assigned:" /tmp/test_peak.log 2>/dev/null || echo "  No data"
echo "First consolidation:"
grep "Unique tokens assigned:" /tmp/test_first.log 2>/dev/null || echo "  No data"
echo "Raw (no gestures):"
grep "Unique tokens assigned:" /tmp/test_raw.log 2>/dev/null || echo "  No data"
