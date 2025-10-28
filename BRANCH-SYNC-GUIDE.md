# Safe Branch Synchronization Guide

## Current Situation

- **Main branch**: Contains older code from earlier development
- **working-on-the-documentation branch**: Contains all recent work including:
  - Gesture token vocabulary implementation
  - Quantizer save/load functionality
  - Dual perception (Wav2Vec + ratio analysis)
  - Visualization system
  - Temporal smoothing
  - Correlation pattern analysis
  - Bug fixes for stuck key issue
  - Documentation updates

These branches have **unrelated histories** (no common ancestor), which is why normal merge fails.

## Recommended Strategy: Safe Reset with Backup

Since `working-on-the-documentation` has all your production code and latest fixes, the safest approach is to make `main` match it.

### Automated Script (Recommended)

I've created a script that does everything safely:

```bash
./sync-branches-safely.sh
```

### Manual Steps (Alternative)

If you prefer to do it manually:

```bash
# 1. Create backup of main
git branch main-backup-$(date +%Y%m%d-%H%M%S) main

# 2. Push working branch to remote
git checkout working-on-the-documentation
git push origin working-on-the-documentation

# 3. Update main to match working branch
git checkout main
git reset --hard working-on-the-documentation

# 4. Push main (requires force since history changed)
git push origin main --force-with-lease

# 5. Return to working branch
git checkout working-on-the-documentation
```

## Why This Is Safe

1. **Creates backup first**: `main-backup-TIMESTAMP` branch preserves old main
2. **Uses --force-with-lease**: Safer than `--force`, won't overwrite if someone else pushed
3. **Preserves working branch**: No changes to your active development branch
4. **Tested and current**: Main will have the production-ready code with all bug fixes

## What Gets Synced

After syncing, `main` will have:

✅ **Training Pipeline**
- 768D Wav2Vec feature extraction
- Gesture token vocabulary (64 tokens)
- Quantizer save/load (`*_quantizer.joblib`)
- Correlation pattern export (`*_correlation_patterns.json`)
- Automatic file naming (saves to `JSON/` directory)

✅ **Live Performance System**
- Dual perception (Wav2Vec + ratio analysis in parallel)
- Brandtsegg rhythm ratio extraction
- Graceful performance fade-out
- Optional live training (`--enable-live-training` flag)
- Distinct behavior modes (SHADOW, MIRROR, COUPLE, etc.)
- Oracle queries with proper gesture token matching

✅ **Visualization System**
- Multi-viewport dashboard
- Real-time pattern matching display
- Request parameters visualization
- Audio analysis viewport
- Performance timeline viewport

✅ **Bug Fixes**
- Stuck key issue (gesture tokens now working)
- Temperature visualization
- Timeline stop mechanism
- Quantizer save for both DualPerceptionModule and HybridPerceptionModule

## Alternative: Keep Both Branches

If you want to preserve main's history separately:

```bash
# Just push working branch and use it as your main development branch
git checkout working-on-the-documentation
git push origin working-on-the-documentation

# Document that working-on-the-documentation is now the primary branch
echo "Primary development branch: working-on-the-documentation" >> README.md
git add README.md
git commit -m "Document primary development branch"
git push origin working-on-the-documentation
```

Then just work on `working-on-the-documentation` and ignore `main`.

## Recovery Plan (If Something Goes Wrong)

If you need to undo the sync:

```bash
# Find your backup branch
git branch | grep main-backup

# Restore main from backup
git checkout main
git reset --hard main-backup-TIMESTAMP
git push origin main --force-with-lease

# Your working branch is untouched, so you're safe
git checkout working-on-the-documentation
```

## Testing After Sync

Once synced, test the system:

```bash
# Train a model
python Chandra_trainer.py --file input_audio/itzama.wav --hybrid-perception --wav2vec --gpu

# Verify files created in JSON/
ls -lh JSON/polyphonic_audio_oracle_training*

# Run live performance
python MusicHal_9000.py --hybrid-perception --wav2vec --gpu --visualize --performance-duration 5
```

All the recent bug fixes should be working! 🎯

