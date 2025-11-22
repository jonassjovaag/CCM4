#!/bin/bash
# Run full sparsity diagnostic with all systems active
# Combines unit tests + timing diagnostic + diversity check

set -e

# Use CCM3 Python directly
PYTHON="CCM3/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ CCM3/bin/python not found"
    exit 1
fi

echo "✅ Using CCM3 Python: $PYTHON"

echo "=============================================================================="
echo "COMPREHENSIVE PHRASE SPARSITY DIAGNOSTIC"
echo "=============================================================================="
echo ""

# Run unit tests first
echo "Step 1: Running unit tests (autonomous state tracking)..."
echo "------------------------------------------------------------------------------"
$PYTHON test_autonomous_pre_somax.py
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed - fix before proceeding"
    exit 1
fi
echo ""

# Run timing diagnostic
echo "Step 2: Running 3-minute timing diagnostic..."
echo "------------------------------------------------------------------------------"
$PYTHON debug_phrase_sparsity.py --duration 3
echo ""

# Run diversity test
echo "Step 3: Checking bass diversity (C2 repetition fix)..."
echo "------------------------------------------------------------------------------"
$PYTHON quick_analysis.py
echo ""

echo "=============================================================================="
echo "DIAGNOSTIC COMPLETE"
echo "=============================================================================="
echo ""
echo "Check logs/ directory for detailed timing events and MIDI output"
echo ""
