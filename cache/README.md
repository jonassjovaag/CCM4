# Event Caching System

This directory contains cached audio event extractions for rapid testing.

## Purpose

Audio extraction (Stage 1) takes ~10 minutes for a full file. By caching extracted events, you can:
1. Extract once (spend the time)
2. Save to cache
3. Reuse for all subsequent test runs while debugging Stages 2-5

**Result**: 10x faster iteration during development.

## Usage

### First Time: Extract and Cache

Extract 1000 events for quick testing:
```bash
python scripts/train/extract_and_cache_events.py \
    input_audio/Bend_like.wav \
    --max-events 1000 \
    --output cache/events/Bend_like_1000.pkl
```

Extract all events for final training:
```bash
python scripts/train/extract_and_cache_events.py \
    input_audio/Bend_like.wav \
    --output cache/events/Bend_like_full.pkl
```

### Subsequent Runs: Use Cached Events

Use cached events to skip extraction:
```bash
python scripts/train/train_modular.py \
    input_audio/Bend_like.wav \
    --cached-events cache/events/Bend_like_1000.pkl \
    --profile quick_test
```

**Time comparison**:
- Without cache: ~590s (extraction) + ~60s (stages 2-5) = ~650s total
- With cache: ~5s (load cache) + ~60s (stages 2-5) = ~65s total
- **Speedup: 10x faster**

## Cache Directory Structure

```
cache/
├── .gitignore          # Prevents committing large .pkl files
├── README.md           # This file
└── events/
    ├── .gitkeep        # Keeps directory in git
    ├── Bend_like_1000.pkl    # 1000 events for testing
    ├── Bend_like_full.pkl    # All events for production
    └── Georgia_1000.pkl      # Other audio files...
```

## When to Re-extract

Delete cache and re-extract when:
- Audio file changes
- You want different event count
- Extraction code changes (new features, bug fixes)

```bash
# Delete specific cache
rm cache/events/Bend_like_1000.pkl

# Delete all caches
rm cache/events/*.pkl

# Then re-extract
python scripts/train/extract_and_cache_events.py ...
```

## Cache File Contents

Each `.pkl` file contains:
```python
{
    'audio_path': 'input_audio/Bend_like.wav',  # Original audio file
    'events': [...],                             # List of AudioEvent objects
    'max_events': 1000,                          # Limit used (None = unlimited)
    'total_events': 1000                         # Actual count
}
```

## Git Configuration

Cache files are **not** tracked by git (too large). The `.gitignore` ensures only the directory structure is committed.

## Tips

1. **Keep multiple caches**: Have both `_1000.pkl` (quick tests) and `_full.pkl` (final training)
2. **Name clearly**: Include event count in filename for easy identification
3. **Check size**: Cache files can be 10-50 MB depending on event count
4. **Invalidate when needed**: Don't forget to re-extract if code changes affect extraction
