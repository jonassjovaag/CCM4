#!/bin/bash
# Test the new gesture → quantizer flow

echo "🧪 Testing NEW Implementation: Gestures → Quantizer → Tokens"
echo "============================================================="
echo ""
echo "This test verifies that consolidation method NOW affects token diversity"
echo ""

source "/Users/jonashsj/Jottacloud/PhD - UiA/CCM4/CCM3/bin/activate"

COMMON_ARGS="--file input_audio/Nineteen.wav \
  --max-events 500 \
  --training-events 100 \
  --no-temporal-smoothing \
  --use-musical-gestures \
  --gesture-transition-threshold 0.5 \
  --gesture-sustain-threshold 0.3 \
  --gesture-min-duration 0.3 \
  --gesture-max-duration 3.0 \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary"

echo "=== TEST 1: Peak Consolidation ===" 
echo "Expected: Should preserve high diversity by selecting most salient moment"
python Chandra_trainer.py \
  --output "JSON/test_new_peak.json" \
  --gesture-consolidation peak \
  $COMMON_ARGS 2>&1 | tee /tmp/test_new_peak.log

echo ""
echo "Peak consolidation results:"
grep "Applying musical gesture consolidation" /tmp/test_new_peak.log | head -1
grep "Consolidated:" /tmp/test_new_peak.log | head -1
grep "Unique tokens assigned:" /tmp/test_new_peak.log | tail -1
echo ""
echo "---"
echo ""

echo "=== TEST 2: Weighted Median Consolidation ==="
echo "Expected: Should show LOWER diversity than peak (averaging effect)"
python Chandra_trainer.py \
  --output "JSON/test_new_weighted.json" \
  --gesture-consolidation weighted_median \
  $COMMON_ARGS 2>&1 | tee /tmp/test_new_weighted.log

echo ""
echo "Weighted median results:"
grep "Applying musical gesture consolidation" /tmp/test_new_weighted.log | head -1
grep "Consolidated:" /tmp/test_new_weighted.log | head -1
grep "Unique tokens assigned:" /tmp/test_new_weighted.log | tail -1
echo ""
echo "---"
echo ""

echo "=== TEST 3: No Gestures (Baseline) ==="
echo "Expected: Highest diversity (no consolidation)"
python Chandra_trainer.py \
  --output "JSON/test_no_gestures.json" \
  --no-musical-gestures \
  --max-events 500 \
  --training-events 100 \
  --no-temporal-smoothing \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary 2>&1 | tee /tmp/test_no_gestures.log

echo ""
echo "No gestures results:"
grep "Training gesture vocabulary" /tmp/test_no_gestures.log | head -1
grep "Unique tokens assigned:" /tmp/test_no_gestures.log | tail -1
echo ""
echo "---"
echo ""

echo "=== FINAL COMPARISON ==="
echo ""
echo "No Gestures (baseline):"
grep "Unique tokens assigned:" /tmp/test_no_gestures.log 2>/dev/null | tail -1 || echo "  Failed"

echo ""
echo "Peak Consolidation:"
grep "Unique tokens assigned:" /tmp/test_new_peak.log 2>/dev/null | tail -1 || echo "  Failed"
grep "Consolidated:" /tmp/test_new_peak.log 2>/dev/null | head -1 || echo "  No consolidation data"

echo ""
echo "Weighted Median:"
grep "Unique tokens assigned:" /tmp/test_new_weighted.log 2>/dev/null | tail -1 || echo "  Failed"
grep "Consolidated:" /tmp/test_new_weighted.log 2>/dev/null | head -1 || echo "  No consolidation data"

echo ""
echo "=== SUCCESS CRITERIA ==="
echo "✅ Weighted median should show FEWER unique tokens than peak"
echo "✅ Both should show consolidation (e.g., 1014 → ~500 segments)"
echo "✅ No gestures should show MOST unique tokens (baseline)"
