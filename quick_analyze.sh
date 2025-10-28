#!/bin/bash
LOG="$1"

if [ -z "$LOG" ]; then
    LOG=$(ls -t logs/conversation_*.csv | head -1)
fi

echo "=== CONVERSATION ANALYSIS ==="
echo "File: $(basename $LOG)"
echo ""

# Count lines
TOTAL=$(tail -n +2 "$LOG" | wc -l | xargs)
INPUTS=$(grep ",INPUT," "$LOG" | wc -l | xargs)
OUTPUTS=$(grep ",OUTPUT," "$LOG" | wc -l | xargs)
MELODY=$(grep ",OUTPUT,melodic," "$LOG" | wc -l | xargs)
BASS=$(grep ",OUTPUT,bass," "$LOG" | wc -l | xargs)

echo "📊 OVERALL:"
echo "   Duration: $(tail -1 "$LOG" | cut -d, -f2) seconds"
echo "   Total events: $TOTAL"
echo "   Human inputs: $INPUTS"
echo "   AI outputs: $OUTPUTS"
echo ""

echo "🎵 AI VOICES:"
echo "   Melodic: $MELODY"
echo "   Bass: $BASS"
echo ""

echo "🎭 MODES (AI):"
for mode in imitate contrast lead; do
    COUNT=$(grep ",OUTPUT,.*,$mode," "$LOG" | wc -l | xargs)
    [ $COUNT -gt 0 ] && echo "   $mode: $COUNT"
done
echo ""

echo "📈 MODE TRANSITIONS:"
AUTO=$(grep ",INPUT,.*,AUTO," "$LOG" | wc -l | xargs)
LISTEN=$(grep ",INPUT,.*,LISTEN," "$LOG" | wc -l | xargs)
echo "   AUTO: $AUTO samples"
echo "   LISTEN: $LISTEN samples"
echo ""

echo "⏱️  TIMELINE:"
grep ",OUTPUT," "$LOG" | while IFS=, read -r ts elapsed type voice pitch note rms activity mode rest; do
    printf "   %6.1fs | %-8s | Note %3s (%-6.1f Hz) | %s\n" "$elapsed" "$voice" "$note" "$pitch" "$mode"
done

