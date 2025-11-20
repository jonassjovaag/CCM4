# TouchOSC Setup Guide for MusicHal 9000 Chord Override

## Quick Start

### 1. Install TouchOSC App
- iOS: Download from App Store
- Android: Download from Google Play
- Cost: ~$5 USD

### 2. Network Setup
**Both devices must be on the same WiFi network!**

1. Find your computer's IP address:
   - macOS: System Settings → Network → WiFi → Details → IP Address
   - Or MusicHal will print it when starting: "Configure TouchOSC to send to: 192.168.x.x:5005"

2. Configure TouchOSC:
   - Open TouchOSC app
   - Tap settings (gear icon)
   - Under "Connections" → "OSC"
   - Set "Host" to your computer's IP (e.g., 192.168.1.100)
   - Set "Send Port" to **5005**
   - Enable "OSC"

### 3. Create Layout

#### Simple Button Layout (Recommended for testing)

Create buttons with these OSC messages:

**Major Chords:**
- C button → `/chord C`
- D button → `/chord D`  
- E button → `/chord E`
- F button → `/chord F`
- G button → `/chord G`
- A button → `/chord A`
- B button → `/chord B`

**Minor Chords:**
- Cm button → `/chord Cm`
- Dm button → `/chord Dm`
- Em button → `/chord Em`
- Fm button → `/chord Fm`
- Gm button → `/chord Gm`
- Am button → `/chord Am`
- Bm button → `/chord Bm`

**Control:**
- CLEAR button → `/chord/clear`

#### Button Configuration in TouchOSC

For each button:
1. Tap button → Edit
2. "OSC" tab
3. Message: `/chord <name>` (e.g., `/chord C`)
4. Value: None (button sends trigger)
5. Save

#### Advanced: Duration Control

Add a rotary/fader to control override duration:
- Message: `/chord/duration`  
- Range: 5-60 seconds
- Value: Float

(Note: Current implementation uses fixed 30s, but you can modify the button messages to include duration as a second argument)

## Usage During Performance

1. **Start MusicHal_9000:**
   ```bash
   python MusicHal_9000.py --enable-rhythmic
   ```
   
2. **Watch for confirmation:**
   ```
   📱 TouchOSC server listening on port 5005
      Configure TouchOSC to send to: 192.168.1.100:5005
      Messages: /chord <name> [duration], /chord/clear
   ```

3. **Test connection** (from another terminal):
   ```bash
   python test_osc_chord_override.py
   ```
   
   Should see in MusicHal console:
   ```
   🎹 MANUAL OVERRIDE: C (for 10s)
      Machine detected: D (conf: 0.24)
      You corrected to: C
   ```

4. **During performance:**
   - Play guitar normally
   - When MusicHal detects wrong chord, tap correct chord on TouchOSC
   - Override lasts 30 seconds (configurable)
   - System plays patterns in YOUR specified chord, not detected chord
   - Status bar shows: `OVERRIDE: C (25s left)`

## Message Format

### Basic Override
```
/chord <chord_name>
```
- Example: `/chord C`, `/chord Dm`, `/chord F#m`
- Duration: 30 seconds (default)

### Override with Custom Duration
```
/chord <chord_name> <duration_seconds>
```
- Example: `/chord C 60` (C major for 60 seconds)
- Example: `/chord Am 15` (A minor for 15 seconds)

### Clear Override
```
/chord/clear
```
- Immediately returns to automatic detection

## Supported Chord Names

**Format:** Root + [modifier]

**Roots:** C, C#, D, D#, E, F, F#, G, G#, A, A#, B  
**Modifiers:** 
- (nothing) = major
- m = minor
- maj7, m7, 7, dim, aug (if your model learned these)

**Examples:**
- `C`, `D`, `E` (majors)
- `Cm`, `Dm`, `Em` (minors)
- `Cmaj7`, `Am7`, `G7` (extended)
- `F#m`, `Bbmaj7` (with accidentals)

## Troubleshooting

### "Connection refused" error
- MusicHal_9000 not running
- Check port 5005 not blocked by firewall

### TouchOSC not working
- Verify same WiFi network
- Check IP address is correct (run `ifconfig` or check MusicHal startup message)
- Try disabling/re-enabling OSC in TouchOSC settings
- Test with `test_osc_chord_override.py` first

### Override not changing sound
- Check console for "🎹 MANUAL OVERRIDE" message
- Verify harmonic progression system enabled (loads transition graph on startup)
- Check status bar shows "OVERRIDE: <chord> (Xs left)"

### Latency/delay
- WiFi latency usually <20ms (imperceptible)
- If laggy: move closer to router, reduce WiFi traffic
- For zero latency: use MIDI controller instead (future feature)

## Research Documentation

All override events are logged to:
- Console: Real-time feedback
- HarmonicContextManager.override_history: In-memory log
- Export with: `harmonic_context_manager.export_override_log("logs/overrides.json")`

Log includes:
- Timestamp
- Detected chord + confidence
- Override chord
- Duration
- Reason (e.g., "TouchOSC manual input")

Perfect for post-performance analysis: "How often did I disagree with the machine?"

## Example TouchOSC Layouts

### Minimal (7 buttons)
```
[C]  [D]  [E]  [F]
[G]  [A]  [B]  [CLEAR]
```

### Full (15 buttons)
```
[C]  [Dm]  [Em]  [F]  [G]
[Am] [Bm]  [F#m][D#m][Gm]
[Cmaj7][Gmaj7][Am7][CLEAR][???]
```

### Performance (includes current chord display)
```
┌─────────────────────────┐
│ Detected: D (24%)       │ ← Label showing detection
│ Active:   C (OVERRIDE)  │ ← Label showing active chord
├─────────────────────────┤
│ [C]  [Dm]  [Em]  [F]    │
│ [G]  [Am]  [F#m][CLEAR] │
│ Duration: [=====|] 30s  │ ← Slider for override duration
└─────────────────────────┘
```

(Note: Current chord display requires custom scripting in TouchOSC - basic buttons work without scripting)

## Next Steps

1. **Test with test_osc_chord_override.py** - Verify OSC communication
2. **Create simple 7-button layout** - Get comfortable with basic chords
3. **Test during improvisation** - See how override feels in practice
4. **Analyze logs** - Study human-machine disagreements for research
5. **Expand layout** - Add chords you use frequently

## Philosophy

This isn't about "fixing wrong detections" - it's about **transparent dialogue**:
- Machine: "I hear D major (24% confident)"
- You: "Actually, I'm playing C major" (tap C button)
- Machine: "Understood - playing patterns in C major for next 30 seconds"

The override system makes the machine's perception **visible and correctable**, building trust through transparency and agency.
