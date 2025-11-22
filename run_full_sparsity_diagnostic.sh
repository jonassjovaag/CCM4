#!/bin/bash
# Run full sparsity diagnostic with all systems active
# Combines unit tests + timing diagnostic + diversity check

set -e

# Activate CCM3 virtual environment
source CCM3/bin/activate
echo "✅ CCM3 virtual environment activated"

echo "=============================================================================="
echo "COMPREHENSIVE PHRASE SPARSITY DIAGNOSTIC"
echo "=============================================================================="
echo ""

# Run unit tests first
echo "Step 1: Running unit tests (autonomous state tracking)..."
echo "------------------------------------------------------------------------------"
python test_autonomous_pre_somax.py
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed - fix before proceeding"
    exit 1
fi
echo ""

# Run timing diagnostic
echo "Step 2: Running 3-minute timing diagnostic..."
echo "------------------------------------------------------------------------------"
python debug_phrase_sparsity.py --duration 3
echo ""

# Run diversity test
echo "Step 3: Checking bass diversity (C2 repetition fix)..."
echo "------------------------------------------------------------------------------"
python quick_analysis.py
echo ""

echo "=============================================================================="
echo "DIAGNOSTIC COMPLETE"
echo "=============================================================================="
echo ""
echo "Check logs/ directory for detailed timing events and MIDI output"
echo ""
