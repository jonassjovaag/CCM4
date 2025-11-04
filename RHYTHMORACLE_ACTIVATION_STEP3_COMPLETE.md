# RhythmOracle Activation - Step 3 of 4 COMPLETE

## Status: Step 3 Completed ✅

**Date**: 2025-11-03  
**Branch**: adding-GPT-OSS-viewport-for-live-usage  
**Objective**: Query RhythmOracle and include rhythmic phrasing in request generation

---

## What Was Done

### 1. Added `_get_rhythmic_phrasing_from_oracle()` Method (Line ~212)

**New method in `agent/phrase_generator.py`:**

```python
def _get_rhythmic_phrasing_from_oracle(self, current_context: Dict = None) -> Optional[Dict]:
    """
    Query RhythmOracle for rhythmic phrasing pattern based on current context
    
    This provides WHEN/HOW to phrase notes (complements AudioOracle's WHAT notes)
    
    Returns:
        Dict with rhythmic phrasing parameters (tempo, density, syncopation, pattern_id)
    """
```

**Functionality**:
- Extracts tempo, density, syncopation from recent human events
- Queries RhythmOracle with `find_similar_patterns(threshold=0.6)`
- Returns best matching pattern with confidence score
- Handles gracefully when RhythmOracle unavailable

**Key Features**:
- **Context building**: Analyzes last 5 human events for rhythmic features
- **Similarity matching**: 60% threshold for pattern matching
- **Confidence scaling**: Pattern confidence × similarity = final confidence
- **Debug logging**: Shows pattern ID, tempo, density, similarity

---

### 2. Integrated Rhythmic Phrasing into Request Building

**Modified three request builders:**

#### Shadow Mode (Line ~278):
```python
# Add rhythmic phrasing if RhythmOracle available
rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
if rhythmic_phrasing:
    request['rhythmic_phrasing'] = rhythmic_phrasing
```

#### Mirror Mode (Line ~356):
```python
# Add rhythmic phrasing if RhythmOracle available  
rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
if rhythmic_phrasing:
    request['rhythmic_phrasing'] = rhythmic_phrasing
```

#### Couple Mode (Line ~422):
```python
# Add rhythmic phrasing if RhythmOracle available
rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
if rhythmic_phrasing:
    request['rhythmic_phrasing'] = rhythmic_phrasing
```

**Effect**: All behavioral modes now include learned rhythmic phrasing patterns in their requests to AudioOracle.

---

## Request Dictionary Structure

Before Step 3:
```python
{
    'harmonic_token': 42,
    'response_mode': 'harmonic',
    'consonance': 0.8,
    'weight': 0.85
}
```

After Step 3:
```python
{
    'harmonic_token': 42,
    'response_mode': 'harmonic',
    'consonance': 0.8,
    'weight': 0.85,
    'rhythmic_phrasing': {  # NEW!
        'tempo': 120.0,
        'density': 0.8,
        'syncopation': 0.3,
        'pattern_type': 'dense',
        'pattern_id': 'pattern_42',
        'confidence': 0.72,  # pattern confidence × similarity
        'meter': '4/4'
    }
}
```

---

## Data Flow Architecture

### Complete Three-System Pipeline:

```
1. YOUR PLAYING (Real-time)
   └─> Brandtsegg Ratio Analysis
       └─> tempo, density, syncopation, IOI ratios
           │
2. RHYTHMIC PHRASING (Pre-trained)
   └─> RhythmOracle.find_similar_patterns()
       └─> rhythmic_phrasing dict
           │
3. HARMONIC CONTENT (Pre-trained)
   └─> AudioOracle.generate_with_request(request + rhythmic_phrasing)
       └─> WHAT notes + WHEN/HOW to phrase them
           │
4. MIDI OUTPUT
   └─> Scheduled notes with learned rhythmic structure
```

### Integration Points:

- **Input**: Human events tracked in `phrase_generator.recent_events`
- **Analysis**: Brandtsegg features extracted per event
- **Query**: RhythmOracle queried with averaged recent features
- **Match**: Similar patterns found via tempo/density/syncopation distance
- **Inject**: `rhythmic_phrasing` dict added to all requests
- **Output**: AudioOracle receives enriched requests with rhythmic context

---

## Progress Tracker

✅ **Step 1**: Initialize RhythmOracle (MusicHal_9000.py line 279)  
✅ **Step 2**: Save/Load RhythmOracle (Chandra_trainer + MusicHal_9000)  
✅ **Step 3**: Generate Rhythmic Decisions (THIS STEP!)  
⏳ **Step 4**: Apply Phrasing to MIDI Output (TODO)

---

## Next: Step 4 - Apply Phrasing to MIDI Output

### Remaining Task:
Currently, the `rhythmic_phrasing` dict is included in the request, but **not yet used to schedule MIDI output timing**.

### What Needs to Happen:

1. **Extract rhythmic phrasing from AudioOracle response**:
   - AudioOracle returns states/patterns with `rhythmic_phrasing` metadata
   - Phrase generator extracts timing parameters

2. **Apply phrasing to note scheduling**:
   - Current: Uniform timing (e.g., 0.5s intervals)
   - Target: Variable timing based on `density`, `syncopation`, `tempo`
   - Use IOI patterns from RhythmOracle to schedule notes

3. **Test with drums input**:
   - Drums → RhythmOracle finds syncopated pattern
   - AudioOracle generates bass notes
   - MIDI output scheduled with syncopated phrasing
   - Result: AI bass line matches drum groove

---

## How It Works (Technical)

### Pattern Matching Algorithm:

```python
# 1. Extract recent rhythmic features
tempos = [120.0, 118.5, 121.0]  # From last 5 events
densities = [0.8, 0.75, 0.82]
syncopations = [0.3, 0.35, 0.28]

# 2. Build query context
query_context = {
    'tempo': mean(tempos) = 119.8,
    'density': mean(densities) = 0.79,
    'syncopation': mean(syncopations) = 0.31
}

# 3. RhythmOracle finds similar patterns
patterns = rhythm_oracle.find_similar_patterns(query_context, threshold=0.6)
# Returns: [(pattern_42, similarity=0.87), (pattern_17, similarity=0.72), ...]

# 4. Use best match
best_pattern = patterns[0]  # pattern_42 with 87% similarity
```

### Similarity Calculation (from `rhythm_oracle.py`):

```python
similarity = (
    tempo_similarity × 0.3 +       # 30% weight
    density_similarity × 0.3 +     # 30% weight
    syncopation_similarity × 0.2 + # 20% weight
    pattern_type_match × 0.2       # 20% weight
)
```

### Confidence Scaling:

```python
final_confidence = pattern.confidence × similarity
# Example: 0.9 (pattern confidence) × 0.87 (similarity) = 0.783
```

---

## Console Output Examples

### When RhythmOracle Active:

```
🥁 RhythmOracle phrasing: pattern=pattern_42, tempo=120.0, density=0.80, similarity=0.87
🎯 Shadow: recent_tokens=[42, 51, 39], consonance=0.75, barlow=5.2
```

### When RhythmOracle Unavailable:

```
🎯 Shadow: recent_tokens=[42, 51, 39], consonance=0.75, barlow=5.2
(No rhythmic phrasing message - silent degradation)
```

### When No Similar Patterns:

```
🥁 RhythmOracle: No similar patterns found for context {'tempo': 180.0, 'density': 0.3, 'syncopation': 0.9}
```

---

## Testing Recommendations

### Test 1: Verify Phrasing Injection

```bash
python MusicHal_9000.py --enable-rhythmic
# Play some notes
# Check console for "🥁 RhythmOracle phrasing: pattern=..." messages
```

**Expected**: Every request should show RhythmOracle query results.

### Test 2: Pattern Matching Quality

Train on varied rhythmic content:
```bash
python Chandra_trainer.py --file varied_rhythms.wav --rhythmic --max-events 10000
```

**Expected**: RhythmOracle learns diverse patterns (sparse, dense, syncopated).

### Test 3: Similarity Threshold

Modify threshold in `_get_rhythmic_phrasing_from_oracle()`:
- `0.5`: More permissive (more matches)
- `0.7`: Stricter (fewer, better matches)
- `0.9`: Very strict (only near-perfect matches)

---

## Known Limitations

### Current Implementation:

1. **High-level patterns only**: Tempo/density/syncopation, not fine-grained IOI sequences
2. **No Factor Oracle structure**: Simple similarity search, not graph traversal
3. **Average-based queries**: Recent events averaged, losing temporal detail
4. **No metric position**: Pattern matching doesn't consider beat position

### Why This Still Works:

- Provides **macro-level rhythmic character** (sparse vs dense, straight vs syncopated)
- Complements AudioOracle's **harmonic specificity**
- Enables **stylistically coherent phrasing** even without micro-timing
- Step 4 will apply these patterns to actual MIDI scheduling

---

## Architectural Notes

### RhythmOracle vs AudioOracle:

| Feature | AudioOracle | RhythmOracle |
|---------|-------------|--------------|
| Structure | Factor Oracle (graph) | Pattern database |
| Learning | Suffix links, transitions | Similarity clustering |
| Queries | Graph traversal | Distance matching |
| Granularity | Event-level (frames) | Pattern-level (sections) |
| Purpose | WHAT notes | WHEN/HOW phrasing |

### Why Different Architectures?

- **AudioOracle**: Needs fine-grained harmonic transitions (note-to-note)
- **RhythmOracle**: Needs macro-level rhythmic character (phrase-level)
- **Together**: Precise harmonic content + coherent rhythmic structure

---

## Success Criteria

✅ **Step 3 Complete When**:
- [x] `_get_rhythmic_phrasing_from_oracle()` method implemented
- [x] All request builders include rhythmic phrasing
- [x] Pattern matching with similarity threshold
- [x] Graceful degradation when unavailable
- [x] Console logging for transparency

🎯 **Overall Success** (After Step 4):
- RhythmOracle patterns applied to MIDI timing
- Drums input → appropriate rhythmic phrasing
- Bass/melody notes scheduled with learned groove
- System demonstrates rhythmic coherence

---

## Commit Message

```
feat: Query RhythmOracle for rhythmic phrasing (Step 3/4)

- Add _get_rhythmic_phrasing_from_oracle() method to PhraseGenerator
- Extract tempo/density/syncopation from recent events
- Query RhythmOracle with find_similar_patterns (threshold=0.6)
- Inject rhythmic_phrasing dict into all request builders
- Shadow/Mirror/Couple modes now include learned rhythmic context

Data flow: 
  Brandtsegg (real-time) → RhythmOracle (query) → 
  AudioOracle (request) → enriched harmonic+rhythmic decisions

Next: Step 4 - Apply phrasing to MIDI output timing
```

---

## Related Files

- `agent/phrase_generator.py` (Lines ~212-275, ~335-350, ~415-430, ~450-465)
- `rhythmic_engine/memory/rhythm_oracle.py` (`find_similar_patterns` method)
- `MusicHal_9000.py` (Lines ~2306-2320, RhythmOracle initialization)

---

**Status**: Ready to proceed with Step 4 (Apply Phrasing to MIDI Output)
