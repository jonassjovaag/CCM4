#!/bin/bash
# Test script to compare token diversity across consolidation methods
# Tests the NEW architecture where gestures are applied BEFORE quantizer training

echo "🎼 Testing Token Diversity with Different Consolidation Methods"
echo "================================================================"
echo ""

# Activate environment
source "/Users/jonashsj/Jottacloud/PhD - UiA/CCM4/CCM3/bin/activate"

# Common parameters
FILE="input_audio/Nineteen.wav"
MAX_EVENTS=500
TRAINING_EVENTS=100

echo "📊 Test Parameters:"
echo "   File: $FILE"
echo "   Max events: $MAX_EVENTS"
echo "   Training events: $TRAINING_EVENTS"
echo "   Architecture: NEW (gestures before quantizer)"
echo ""

# Test 1: Peak consolidation (selection-based)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 TEST 1: Peak Consolidation (select most salient)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python Chandra_trainer.py \
  --file "$FILE" \
  --output "JSON/test_peak.json" \
  --max-events $MAX_EVENTS \
  --training-events $TRAINING_EVENTS \
  --no-temporal-smoothing \
  --use-musical-gestures \
  --gesture-consolidation peak \
  --gesture-transition-threshold 0.5 \
  --gesture-sustain-threshold 0.3 \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary 2>&1 | grep -E "(Unique tokens|Active tokens|Consolidated:|Entropy:)"

echo ""
echo ""

# Test 2: First consolidation (onset-based)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 TEST 2: First Consolidation (use gesture onset)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python Chandra_trainer.py \
  --file "$FILE" \
  --output "JSON/test_first.json" \
  --max-events $MAX_EVENTS \
  --training-events $TRAINING_EVENTS \
  --no-temporal-smoothing \
  --use-musical-gestures \
  --gesture-consolidation first \
  --gesture-transition-threshold 0.5 \
  --gesture-sustain-threshold 0.3 \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary 2>&1 | grep -E "(Unique tokens|Active tokens|Consolidated:|Entropy:)"

echo ""
echo ""

# Test 3: Weighted median (the problematic averaging)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 TEST 3: Weighted Median (averaging - expected lower diversity)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python Chandra_trainer.py \
  --file "$FILE" \
  --output "JSON/test_weighted_median.json" \
  --max-events $MAX_EVENTS \
  --training-events $TRAINING_EVENTS \
  --no-temporal-smoothing \
  --use-musical-gestures \
  --gesture-consolidation weighted_median \
  --gesture-transition-threshold 0.5 \
  --gesture-sustain-threshold 0.3 \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary 2>&1 | grep -E "(Unique tokens|Active tokens|Consolidated:|Entropy:)"

echo ""
echo ""

# Test 4: No gestures (baseline - maximum diversity)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 TEST 4: No Gestures (baseline - maximum diversity)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python Chandra_trainer.py \
  --file "$FILE" \
  --output "JSON/test_no_gestures.json" \
  --max-events $MAX_EVENTS \
  --training-events $TRAINING_EVENTS \
  --no-temporal-smoothing \
  --no-hierarchical \
  --no-rhythmic \
  --no-gpt-oss \
  --no-dual-vocabulary 2>&1 | grep -E "(Unique tokens|Active tokens|Entropy:)"

echo ""
echo ""
echo "================================================================"
echo "🎯 SUMMARY OF RESULTS"
echo "================================================================"
echo ""
echo "Expected outcomes:"
echo "  • No Gestures: ~55-64 unique tokens (baseline)"
echo "  • Peak: ~45-55 tokens (slight reduction from consolidation)"
echo "  • First: ~45-55 tokens (similar to peak)"
echo "  • Weighted Median: ~30-45 tokens (more reduction but NOT 1!)"
echo ""
echo "✅ SUCCESS CRITERIA:"
echo "   1. All methods > 20 unique tokens (no catastrophic collapse)"
echo "   2. Peak ≥ Weighted Median (selection preserves more diversity)"
echo "   3. No gestures > any consolidation method (baseline highest)"
echo ""
