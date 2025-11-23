#!/bin/bash
# Test script for RhythmOracle rhythmic pattern integration
# Tests if syncopation from Brandtsegg ratio analysis is applied to live MIDI output

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}RhythmOracle Integration Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create test logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/rhythmic_integration_test_${TIMESTAMP}.log"
ANALYSIS_FILE="$LOG_DIR/rhythmic_analysis_${TIMESTAMP}.txt"
MIDI_CSV="logs/midi_output_${TIMESTAMP}.csv"

echo -e "${YELLOW}Log files:${NC}"
echo "  Console output: $LOG_FILE"
echo "  Analysis report: $ANALYSIS_FILE"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating CCM3 virtual environment...${NC}"
source CCM3/bin/activate

# Check if Moon_stars.json model exists
if [ ! -f "JSON/Moon_stars.json" ]; then
    echo -e "${RED}ERROR: Moon_stars.json model not found!${NC}"
    echo "Please train the model first with:"
    echo "  python Chandra_trainer.py input_audio/Moon_stars.wav --max-events 8000 --training-events 2000"
    exit 1
fi

echo -e "${GREEN}✓${NC} Model found: JSON/Moon_stars.json"
echo ""

# Run 2-minute test performance
echo -e "${YELLOW}Running 2-minute test performance...${NC}"
echo "  Command: python MusicHal_9000.py --performance-duration 2 --enable-meld"
echo "  Output: $LOG_FILE"
echo ""

# Run with output capture
python MusicHal_9000.py --performance-duration 2 --enable-meld 2>&1 | tee "$LOG_FILE"

TEST_EXIT_CODE=${PIPESTATUS[0]}

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}ERROR: Performance test failed with exit code $TEST_EXIT_CODE${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓${NC} Performance completed successfully"
echo ""

# Wait a moment for logs to flush
sleep 2

# Find the most recent MIDI output CSV
LATEST_MIDI_CSV=$(ls -t logs/midi_output_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_MIDI_CSV" ]; then
    echo -e "${RED}ERROR: No MIDI output CSV found in logs/!${NC}"
    exit 1
fi

echo -e "${YELLOW}Analyzing MIDI output: $LATEST_MIDI_CSV${NC}"
echo ""

# Analyze rhythmic patterns using Python
python3 << EOF
import csv
import numpy as np
from collections import Counter
import sys

# Read MIDI output
note_on_times = []
with open('$LATEST_MIDI_CSV', 'r') as f:
    next(f)  # Skip comment line
    reader = csv.DictReader(f)
    for row in reader:
        if row['message_type'] == 'note_on':
            note_on_times.append(float(row['timestamp']))

if len(note_on_times) < 5:
    print("ERROR: Not enough MIDI notes generated (< 5)")
    sys.exit(1)

# Calculate inter-onset intervals
iois = np.diff(note_on_times)

if len(iois) < 3:
    print("ERROR: Not enough inter-onset intervals (< 3)")
    sys.exit(1)

# Statistical analysis
ioi_mean = np.mean(iois)
ioi_std = np.std(iois)
ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0  # Coefficient of variation

# Check for syncopation (off-grid timing)
# Assume 120 BPM as base (0.5s per beat)
beat_duration = 0.5
grid_positions = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # Common grid points

off_grid_count = 0
on_grid_count = 0

for ioi in iois:
    # Check if IOI is close to any grid position (±10%)
    is_on_grid = False
    for grid_pos in grid_positions:
        if abs(ioi - grid_pos) < (grid_pos * 0.1):
            is_on_grid = True
            break
    
    if is_on_grid:
        on_grid_count += 1
    else:
        off_grid_count += 1

syncopation_ratio = off_grid_count / len(iois) if len(iois) > 0 else 0

# Check IOI distribution (should NOT be uniform random)
# Bin IOIs and check for non-uniform distribution
ioi_bins = np.histogram(iois, bins=10)[0]
ioi_entropy = -np.sum((ioi_bins / len(iois)) * np.log2((ioi_bins / len(iois)) + 1e-10))
max_entropy = np.log2(10)  # Maximum entropy for 10 bins
normalized_entropy = ioi_entropy / max_entropy

# Write analysis report
with open('$ANALYSIS_FILE', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("RhythmOracle Integration Analysis Report\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Test timestamp: $TIMESTAMP\n")
    f.write(f"MIDI file analyzed: $LATEST_MIDI_CSV\n")
    f.write(f"Total notes: {len(note_on_times)}\n")
    f.write(f"Inter-onset intervals: {len(iois)}\n\n")
    
    f.write("-" * 60 + "\n")
    f.write("Inter-Onset Interval Statistics\n")
    f.write("-" * 60 + "\n")
    f.write(f"Mean IOI: {ioi_mean:.3f}s\n")
    f.write(f"Std Dev IOI: {ioi_std:.3f}s\n")
    f.write(f"Coefficient of Variation: {ioi_cv:.3f}\n")
    f.write(f"Min IOI: {np.min(iois):.3f}s\n")
    f.write(f"Max IOI: {np.max(iois):.3f}s\n")
    f.write(f"Median IOI: {np.median(iois):.3f}s\n\n")
    
    f.write("-" * 60 + "\n")
    f.write("Syncopation Analysis\n")
    f.write("-" * 60 + "\n")
    f.write(f"On-grid IOIs: {on_grid_count} ({on_grid_count/len(iois)*100:.1f}%)\n")
    f.write(f"Off-grid IOIs: {off_grid_count} ({off_grid_count/len(iois)*100:.1f}%)\n")
    f.write(f"Syncopation ratio: {syncopation_ratio:.3f}\n\n")
    
    f.write("-" * 60 + "\n")
    f.write("Pattern Complexity\n")
    f.write("-" * 60 + "\n")
    f.write(f"Normalized entropy: {normalized_entropy:.3f}\n")
    f.write(f"  (0.0 = all same timing, 1.0 = completely random)\n\n")
    
    # Check for RhythmOracle usage in logs
    rhythm_oracle_mentions = 0
    fallback_mentions = 0
    with open('$LOG_FILE', 'r') as log:
        for line in log:
            if '🥁 RhythmOracle: Applied pattern' in line:
                rhythm_oracle_mentions += 1
            elif '🥁' in line and 'No RhythmOracle patterns found' in line:
                fallback_mentions += 1
    
    f.write("-" * 60 + "\n")
    f.write("RhythmOracle Usage\n")
    f.write("-" * 60 + "\n")
    f.write(f"Patterns applied: {rhythm_oracle_mentions}\n")
    f.write(f"Fallback timing used: {fallback_mentions}\n")
    
    total_decisions = rhythm_oracle_mentions + fallback_mentions
    if total_decisions > 0:
        usage_pct = rhythm_oracle_mentions / total_decisions * 100
        f.write(f"RhythmOracle usage: {usage_pct:.1f}%\n\n")
    else:
        f.write("RhythmOracle usage: Unknown (no debug logs found)\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("Test Results\n")
    f.write("=" * 60 + "\n\n")
    
    # Determine if test passes
    test_passed = True
    reasons = []
    
    # Check 1: Syncopation detected (> 10% off-grid)
    if syncopation_ratio < 0.1:
        test_passed = False
        reasons.append(f"Low syncopation ({syncopation_ratio:.1%} off-grid, need >10%)")
    else:
        f.write(f"✓ Syncopation detected: {syncopation_ratio:.1%} off-grid timing\n")
    
    # Check 2: Non-uniform distribution (entropy between 0.3-0.9)
    if normalized_entropy < 0.3 or normalized_entropy > 0.9:
        test_passed = False
        reasons.append(f"Timing distribution issue (entropy={normalized_entropy:.2f}, expect 0.3-0.9)")
    else:
        f.write(f"✓ Non-uniform timing: entropy={normalized_entropy:.2f}\n")
    
    # Check 3: RhythmOracle usage > 50%
    if total_decisions > 0 and usage_pct < 50:
        test_passed = False
        reasons.append(f"Low RhythmOracle usage ({usage_pct:.1f}%, need >50%)")
    elif total_decisions > 0:
        f.write(f"✓ RhythmOracle usage: {usage_pct:.1f}%\n")
    
    # Check 4: Reasonable IOI variance
    if ioi_cv < 0.15:
        test_passed = False
        reasons.append(f"Low timing variance (CV={ioi_cv:.2f}, expect >0.15)")
    else:
        f.write(f"✓ Timing variance: CV={ioi_cv:.2f}\n")
    
    f.write("\n")
    
    if test_passed:
        f.write("🎉 TEST PASSED: Rhythmic patterns successfully integrated!\n")
        print("\n" + "=" * 60)
        print("🎉 TEST PASSED")
        print("=" * 60)
        print(f"Syncopation ratio: {syncopation_ratio:.1%}")
        print(f"Timing entropy: {normalized_entropy:.2f}")
        if total_decisions > 0:
            print(f"RhythmOracle usage: {usage_pct:.1f}%")
        print(f"\nFull report: $ANALYSIS_FILE")
        sys.exit(0)
    else:
        f.write("❌ TEST FAILED\n\n")
        f.write("Reasons:\n")
        for reason in reasons:
            f.write(f"  - {reason}\n")
        
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        for reason in reasons:
            print(f"  - {reason}")
        print(f"\nFull report: $ANALYSIS_FILE")
        print(f"Console log: $LOG_FILE")
        sys.exit(1)

EOF

ANALYSIS_EXIT_CODE=$?

echo ""
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Tests failed - see analysis above${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
