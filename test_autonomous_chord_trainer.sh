#!/bin/bash
# Quick test script for autonomous chord trainer

echo "🎵 Autonomous Chord Trainer - Quick Test"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Test MIDI output connection"
echo "  2. Test audio input"
echo "  3. Train on 3 chords as a demo"
echo ""
echo "Press Ctrl+C to stop at any time"
echo ""

# Check if IAC Driver exists
python3 -c "
import mido
ports = mido.get_output_names()
print('📡 Available MIDI ports:')
for p in ports:
    print(f'   - {p}')

if 'IAC Driver Chord Trainer Output' in ports:
    print('\n✅ IAC Driver Chord Trainer Output found!')
else:
    print('\n❌ IAC Driver Chord Trainer Output NOT found!')
    print('💡 Please create it in Audio MIDI Setup')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup required. See AUTONOMOUS_CHORD_TRAINER_SETUP.md"
    exit 1
fi

echo ""
echo "🎤 Testing audio input (device 2 - MacBook microphone)..."
python3 test_audio_devices.py | grep "MacBook Pro Microphone" -A 3

echo ""
echo "🚀 Ready to start! Make sure Ableton is set up:"
echo "  ✓ Track armed with 'Chord Trainer Output' as input"
echo "  ✓ Piano/synth loaded"
echo "  ✓ Volume up"
echo ""
read -p "Press ENTER to start training (Ctrl+C to cancel)..."

# Run quick test with 3 chords only
echo ""
echo "Starting quick test..."
python3 autonomous_chord_trainer.py --input-device 2 --chord-duration 1.5 --train-interval 10

echo ""
echo "✅ Test complete! Check models/ directory for output files."































