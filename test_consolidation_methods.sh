#!/bin/bash
# Test consolidation methods with parameters that actually create multi-event gestures

echo "🧪 Testing Consolidation Methods (with actual multi-event gestures)"
echo "===================================================================="
echo ""
echo "Goal: Create gestures with multiple events, then compare consolidation methods"
echo ""

# Parameters that encourage consolidation:
# - Higher transition threshold (0.6) = less sensitive to changes
# - Higher sustain threshold (0.4) = needs more similarity to end gesture
# - Longer max duration (5.0s) = allows longer gestures
# - Longer min duration (0.5s) = forces events to group

COMMON_ARGS="--file input_audio/Nineteen.wav \
  --max-events 500 \
  --training-events 100 \
  --no-temporal-smoothing \
  --use-musical-gestures \
  --gesture-transition-threshold 0.6 \
  --gesture-sustain-threshold 0.4 \
  --gesture-min-duration 0.5 \
  --gesture-max-duration 5.0 \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary"

echo "=== TEST 1: Peak Consolidation (select most salient moment) ==="
echo "Expected: High diversity - selects most prominent event in each gesture"
source "/Users/jonashsj/Jottacloud/PhD - UiA/CCM4/CCM3/bin/activate" && \
python Chandra_trainer.py \
  --output "JSON/test_consolidation_peak.json" \
  --gesture-consolidation peak \
  $COMMON_ARGS 2>&1 | tee /tmp/test_peak.log | grep -A5 "Musical Gesture Processing Statistics"

echo ""
echo "Peak result:"
grep "Unique tokens assigned:" /tmp/test_peak.log || echo "❌ No token data"
grep "Gestures created:" /tmp/test_peak.log | head -1 || echo "❌ No gesture data"
grep "Consolidation ratio:" /tmp/test_peak.log | head -1 || echo "❌ No ratio data"
echo ""
echo "---"
echo ""

echo "=== TEST 2: First Consolidation (select gesture onset) ==="
echo "Expected: High diversity - selects first event in each gesture"
source "/Users/jonashsj/Jottacloud/PhD - UiA/CCM4/CCM3/bin/activate" && \
python Chandra_trainer.py \
  --output "JSON/test_consolidation_first.json" \
  --gesture-consolidation first \
  $COMMON_ARGS 2>&1 | tee /tmp/test_first.log | grep -A5 "Musical Gesture Processing Statistics"

echo ""
echo "First result:"
grep "Unique tokens assigned:" /tmp/test_first.log || echo "❌ No token data"
grep "Gestures created:" /tmp/test_first.log | head -1 || echo "❌ No gesture data"
grep "Consolidation ratio:" /tmp/test_first.log | head -1 || echo "❌ No ratio data"
echo ""
echo "---"
echo ""

echo "=== TEST 3: Weighted Median Consolidation (ORIGINAL - averages features) ==="
echo "Expected: LOW diversity - averaging destroys distinctiveness"
source "/Users/jonashsj/Jottacloud/PhD - UiA/CCM4/CCM3/bin/activate" && \
python Chandra_trainer.py \
  --output "JSON/test_consolidation_weighted_median.json" \
  --gesture-consolidation weighted_median \
  $COMMON_ARGS 2>&1 | tee /tmp/test_weighted.log | grep -A5 "Musical Gesture Processing Statistics"

echo ""
echo "Weighted median result:"
grep "Unique tokens assigned:" /tmp/test_weighted.log || echo "❌ No token data"
grep "Gestures created:" /tmp/test_weighted.log | head -1 || echo "❌ No gesture data"
grep "Consolidation ratio:" /tmp/test_weighted.log | head -1 || echo "❌ No ratio data"
echo ""
echo "---"
echo ""

echo "=== SUMMARY COMPARISON ==="
echo ""
echo "Peak consolidation (select most salient):"
grep "Unique tokens assigned:" /tmp/test_peak.log 2>/dev/null | tail -1 || echo "  No data"
grep "Consolidation ratio:" /tmp/test_peak.log 2>/dev/null | head -1 || echo "  No consolidation data"
echo ""
echo "First consolidation (select gesture onset):"
grep "Unique tokens assigned:" /tmp/test_first.log 2>/dev/null | tail -1 || echo "  No data"
grep "Consolidation ratio:" /tmp/test_first.log 2>/dev/null | head -1 || echo "  No consolidation data"
echo ""
echo "Weighted median (ORIGINAL - averages features):"
grep "Unique tokens assigned:" /tmp/test_weighted.log 2>/dev/null | tail -1 || echo "  No data"
grep "Consolidation ratio:" /tmp/test_weighted.log 2>/dev/null | head -1 || echo "  No consolidation data"
echo ""
echo "=== HYPOTHESIS ==="
echo "✅ Peak/First should show HIGH unique token counts (selection preserves diversity)"
echo "❌ Weighted median should show LOW unique token counts (averaging destroys diversity)"
echo ""
echo "If consolidation ratio is 1.00x for all, parameters need more adjustment."
