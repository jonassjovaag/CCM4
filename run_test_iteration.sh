#!/bin/bash
# Quick test iteration script
# Runs MusicHal_9000 for 10 seconds, then analyzes logs

echo "🧪 ITERATION TEST - 10 seconds"
echo "================================"

# Start MusicHal_9000 in background
timeout 10 python MusicHal_9000.py &
MUSIC_PID=$!

# Start synthetic audio after 1 second delay
sleep 1
timeout 10 python test_synthetic_audio.py --duration 10 &
AUDIO_PID=$!

# Wait for both to finish
wait $MUSIC_PID
wait $AUDIO_PID

echo ""
echo "✅ Test complete! Analyzing logs..."
echo ""

# Get most recent log files
LATEST_CONV=$(ls -t logs/conversation_*.csv 2>/dev/null | head -1)
LATEST_TERM=$(ls -t logs/terminal_*.log 2>/dev/null | head -1)

if [ -f "$LATEST_CONV" ]; then
    echo "📊 Conversation Log: $LATEST_CONV"
    echo "---"
    echo "Total lines: $(wc -l < "$LATEST_CONV")"
    echo "INPUT events: $(grep -c "INPUT" "$LATEST_CONV")"
    echo "OUTPUT melodic: $(grep -c "OUTPUT,melodic" "$LATEST_CONV")"
    echo "OUTPUT bass: $(grep -c "OUTPUT,bass" "$LATEST_CONV")"
    echo ""
    echo "First 10 OUTPUT events:"
    grep "OUTPUT" "$LATEST_CONV" | head -10
fi

echo ""
echo "🔍 Check full logs:"
echo "  Terminal: $LATEST_TERM"
echo "  Conversation: $LATEST_CONV"

