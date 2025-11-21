#!/usr/bin/env python3
"""
Diagnostic script to identify performance issues in MusicHal 9000.

Checks:
1. Token collapse - Are all gestures mapping to the same token?
2. AudioOracle frame availability - Can interval extraction find frames?
3. RhythmOracle pattern matching - Why do queries return 0 patterns?
"""

import json
import joblib
import numpy as np
from pathlib import Path

print("=" * 80)
print("MusicHal 9000 Performance Diagnostics")
print("=" * 80)

# ==============================================================================
# PROBLEM 1: Token Collapse - Vocabulary Health Check
# ==============================================================================
print("\n[1/3] CHECKING VOCABULARY HEALTH (Token Collapse Detection)")
print("-" * 80)

try:
    harmonic_vocab_path = "input_audio/General_idea_harmonic_vocab.joblib"
    percussive_vocab_path = "input_audio/General_idea_percussive_vocab.joblib"
    
    if Path(harmonic_vocab_path).exists():
        print(f"Loading harmonic vocabulary: {harmonic_vocab_path}")
        harmonic_vocab = joblib.load(harmonic_vocab_path)
        
        # Check structure
        print(f"  ✓ Vocabulary size: {harmonic_vocab.get('vocabulary_size', 'unknown')}")
        print(f"  ✓ Feature dimension: {harmonic_vocab.get('feature_dim', 'unknown')}")
        print(f"  ✓ Codebook shape: {harmonic_vocab['codebook'].shape}")
        print(f"  ✓ Use PCA: {harmonic_vocab.get('use_pca', False)}")
        if harmonic_vocab.get('use_pca'):
            print(f"  ✓ PCA: {harmonic_vocab['feature_dim']}D → {harmonic_vocab.get('reduced_dim', 'unknown')}D")
        
        # Check for token collapse
        codebook = harmonic_vocab['codebook']
        kmeans = harmonic_vocab['kmeans']
        
        # Predict which cluster each codebook vector belongs to
        token_assignments = kmeans.predict(codebook)
        unique_tokens = len(set(token_assignments))
        
        print(f"\n  Token diversity check:")
        print(f"  ✓ Expected unique tokens: {harmonic_vocab['vocabulary_size']}")
        print(f"  ✓ Actual unique tokens: {unique_tokens}")
        
        if unique_tokens == 1:
            print(f"  ❌ CRITICAL: All codebook vectors map to token {token_assignments[0]}!")
            print(f"     This is TOTAL TOKEN COLLAPSE - vocabulary is useless!")
        elif unique_tokens < harmonic_vocab['vocabulary_size'] * 0.5:
            print(f"  ⚠️  WARNING: Only {unique_tokens}/{harmonic_vocab['vocabulary_size']} tokens active")
            print(f"     Vocabulary has low diversity - may need retraining")
        else:
            print(f"  ✓ Good diversity: {unique_tokens}/{harmonic_vocab['vocabulary_size']} tokens active")
        
        # Check token frequency distribution
        if 'token_frequencies' in harmonic_vocab:
            freqs = harmonic_vocab['token_frequencies']
            print(f"\n  Token frequency distribution:")
            print(f"  ✓ Total tokens tracked: {len(freqs)}")
            sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            print(f"  ✓ Top 5 most frequent tokens:")
            for token, count in sorted_freqs[:5]:
                pct = (count / sum(freqs.values())) * 100
                print(f"     Token {token}: {count} occurrences ({pct:.1f}%)")
            
            # Check if one token dominates
            max_freq = max(freqs.values())
            total_freq = sum(freqs.values())
            dominance = (max_freq / total_freq) * 100
            if dominance > 50:
                print(f"  ⚠️  WARNING: Token {sorted_freqs[0][0]} dominates with {dominance:.1f}% of all occurrences!")
        
        # Check cluster separation
        print(f"\n  Cluster separation check:")
        cluster_centers = kmeans.cluster_centers_
        if len(cluster_centers) > 1:
            # Compute pairwise distances between cluster centers
            from scipy.spatial.distance import pdist
            distances = pdist(cluster_centers, metric='euclidean')
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            mean_distance = np.mean(distances)
            
            print(f"  ✓ Min cluster separation: {min_distance:.4f}")
            print(f"  ✓ Max cluster separation: {max_distance:.4f}")
            print(f"  ✓ Mean cluster separation: {mean_distance:.4f}")
            
            if min_distance < 0.01:
                print(f"  ❌ CRITICAL: Clusters are too close (min={min_distance:.4f})!")
                print(f"     Some tokens are indistinguishable - vocabulary needs retraining")
            elif min_distance < harmonic_vocab.get('min_cluster_separation', 0.1):
                print(f"  ⚠️  WARNING: Cluster separation below threshold")
    else:
        print(f"  ❌ Harmonic vocabulary not found: {harmonic_vocab_path}")
    
    # Check percussive vocabulary
    if Path(percussive_vocab_path).exists():
        print(f"\n  Percussive vocabulary: {percussive_vocab_path}")
        percussive_vocab = joblib.load(percussive_vocab_path)
        print(f"  ✓ Vocabulary size: {percussive_vocab.get('vocabulary_size', 'unknown')}")
        
        # Quick token collapse check
        codebook = percussive_vocab['codebook']
        kmeans = percussive_vocab['kmeans']
        token_assignments = kmeans.predict(codebook)
        unique_tokens = len(set(token_assignments))
        print(f"  ✓ Unique tokens: {unique_tokens}/{percussive_vocab['vocabulary_size']}")
        
        if unique_tokens == 1:
            print(f"  ❌ CRITICAL: Percussive vocabulary also collapsed!")

except Exception as e:
    print(f"  ❌ Error checking vocabulary: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# PROBLEM 2: AudioOracle Frame Availability
# ==============================================================================
print("\n[2/3] CHECKING AUDIOORACLE FRAME AVAILABILITY")
print("-" * 80)

try:
    model_path = "JSON/General_idea.json"
    
    if Path(model_path).exists():
        print(f"Loading AudioOracle model: {model_path}")
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Check structure
        if 'data' in model_data and 'audio_oracle' in model_data['data']:
            oracle_data = model_data['data']['audio_oracle']
            
            print(f"  ✓ AudioOracle structure found")
            print(f"  ✓ States: {len(oracle_data.get('states', {}))}")
            print(f"  ✓ Transitions: {len(oracle_data.get('transitions', {}))}")
            print(f"  ✓ Suffix links: {len(oracle_data.get('suffix_links', {}))}")
            
            # Check audio_frames
            audio_frames = oracle_data.get('audio_frames', {})
            print(f"\n  Audio frames check:")
            print(f"  ✓ Total frames: {len(audio_frames)}")
            
            if len(audio_frames) > 0:
                # Sample frame structure
                sample_frame_id = list(audio_frames.keys())[0]
                sample_frame = audio_frames[sample_frame_id]
                
                print(f"  ✓ Sample frame ID: {sample_frame_id}")
                print(f"  ✓ Sample frame keys: {list(sample_frame.keys())}")
                
                # Check if frame has required data for interval extraction
                if 'features' in sample_frame:
                    features = sample_frame['features']
                    print(f"  ✓ Features present: {len(features)}D")
                else:
                    print(f"  ❌ WARNING: Sample frame missing 'features' key!")
                
                if 'audio_data' in sample_frame:
                    audio_data = sample_frame['audio_data']
                    print(f"  ✓ Audio data present: {type(audio_data)}")
                    if isinstance(audio_data, dict):
                        print(f"     Keys: {list(audio_data.keys())[:5]}...")
                        # Check for MIDI/pitch information
                        if 'midi' in audio_data:
                            print(f"     ✓ MIDI data available")
                        else:
                            print(f"     ⚠️  WARNING: No 'midi' key in audio_data!")
                            print(f"        Interval extraction may fail!")
                else:
                    print(f"  ❌ WARNING: Sample frame missing 'audio_data' key!")
                
                # Check frame ID format
                print(f"\n  Frame ID format check:")
                frame_ids = list(audio_frames.keys())
                print(f"  ✓ First 10 frame IDs: {frame_ids[:10]}")
                
                # Check if frame IDs are sequential or have gaps
                try:
                    frame_id_ints = [int(fid) for fid in frame_ids]
                    print(f"  ✓ Frame IDs are numeric: {min(frame_id_ints)} to {max(frame_id_ints)}")
                    
                    # Check for gaps
                    expected_ids = set(range(min(frame_id_ints), max(frame_id_ints) + 1))
                    actual_ids = set(frame_id_ints)
                    missing_ids = expected_ids - actual_ids
                    if missing_ids:
                        print(f"  ⚠️  WARNING: {len(missing_ids)} missing frame IDs in sequence")
                        if len(missing_ids) <= 10:
                            print(f"     Missing: {sorted(list(missing_ids))}")
                except:
                    print(f"  ✓ Frame IDs are not purely numeric (may use string IDs)")
                
            else:
                print(f"  ❌ CRITICAL: No audio frames found!")
                print(f"     Interval extraction will always fail!")
        else:
            print(f"  ❌ AudioOracle data structure not found in model")
    else:
        print(f"  ❌ Model file not found: {model_path}")

except Exception as e:
    print(f"  ❌ Error checking AudioOracle: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# PROBLEM 3: RhythmOracle Pattern Matching
# ==============================================================================
print("\n[3/3] CHECKING RHYTHMORACLE PATTERN MATCHING")
print("-" * 80)

try:
    # Load RhythmOracle patterns from model
    if Path(model_path).exists():
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        if 'data' in model_data and 'rhythm_oracle' in model_data['data']:
            rhythm_data = model_data['data']['rhythm_oracle']
            patterns = rhythm_data.get('patterns', [])
            
            print(f"  ✓ RhythmOracle patterns loaded: {len(patterns)}")
            
            if len(patterns) > 0:
                print(f"\n  Trained pattern analysis:")
                for i, pattern in enumerate(patterns):
                    print(f"\n  Pattern {i} ({pattern.get('pattern_id', 'unknown')}):")
                    print(f"    Duration pattern: {pattern.get('duration_pattern', 'N/A')}")
                    print(f"    Density: {pattern.get('density', 'N/A'):.2f} onsets/sec")
                    print(f"    Syncopation: {pattern.get('syncopation', 'N/A'):.2f}")
                    print(f"    Pulse: {pattern.get('pulse', 'N/A')}")
                    print(f"    Meter: {pattern.get('meter', 'N/A')}")
                    print(f"    Pattern type: {pattern.get('pattern_type', 'N/A')}")
                    print(f"    Confidence: {pattern.get('confidence', 'N/A'):.2f}")
                
                # Simulate query patterns from log
                print(f"\n  Simulated query patterns (from log):")
                query_patterns = [
                    {'duration_pattern': [1, 2, 1, 1], 'density': 0.5, 'syncopation': 0.0, 'pulse': 4},
                    {'duration_pattern': [2, 1, 1, 1], 'density': 0.5, 'syncopation': 0.0, 'pulse': 4},
                    {'duration_pattern': [2, 2, 2, 2], 'density': 0.5, 'syncopation': 0.0, 'pulse': 4},
                ]
                
                for j, query in enumerate(query_patterns):
                    print(f"\n  Query {j + 1}: {query}")
                    print(f"    Comparing against trained patterns...")
                    
                    # Simple similarity check
                    for i, pattern in enumerate(patterns):
                        # Duration pattern similarity (normalized)
                        query_dur = np.array(query['duration_pattern'])
                        pattern_dur = np.array(pattern.get('duration_pattern', []))
                        
                        if len(pattern_dur) > 0:
                            # Normalize and compare
                            query_norm = query_dur / np.sum(query_dur) if np.sum(query_dur) > 0 else query_dur
                            pattern_norm = pattern_dur / np.sum(pattern_dur) if np.sum(pattern_dur) > 0 else pattern_dur
                            
                            # Pad to same length
                            max_len = max(len(query_norm), len(pattern_norm))
                            query_padded = np.pad(query_norm, (0, max_len - len(query_norm)))
                            pattern_padded = np.pad(pattern_norm, (0, max_len - len(pattern_norm)))
                            
                            dur_similarity = 1.0 - np.mean(np.abs(query_padded - pattern_padded))
                            
                            # Density similarity
                            density_diff = abs(query['density'] - pattern.get('density', 0.5))
                            density_similarity = 1.0 - min(density_diff / 5.0, 1.0)  # Normalize by max expected density
                            
                            # Overall similarity
                            overall_sim = (dur_similarity * 0.7 + density_similarity * 0.3)
                            
                            print(f"      vs Pattern {i}: duration_sim={dur_similarity:.2f}, density_sim={density_similarity:.2f}, overall={overall_sim:.2f}")
                        else:
                            print(f"      vs Pattern {i}: No duration pattern available")
                    
                # Identify the problem
                print(f"\n  Analysis:")
                densities = [p.get('density', 0.5) for p in patterns]
                print(f"    Trained pattern density range: {min(densities):.2f} - {max(densities):.2f} onsets/sec")
                print(f"    Query pattern density: 0.5 onsets/sec")
                print(f"    Density mismatch: {abs(min(densities) - 0.5):.2f}x difference")
                
                if min(densities) > 2.0:
                    print(f"\n  ❌ PROBLEM IDENTIFIED:")
                    print(f"     Trained patterns have MUCH higher density ({min(densities):.1f}+) than")
                    print(f"     query patterns (0.5). This is why similarity matching fails!")
                    print(f"     Solution: Either retrain with sparser patterns OR adjust query density")
                    print(f"     calculation to match trained data scale.")
            else:
                print(f"  ❌ CRITICAL: No rhythmic patterns found in model!")
        else:
            print(f"  ❌ RhythmOracle data not found in model")
    
except Exception as e:
    print(f"  ❌ Error checking RhythmOracle: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# SUMMARY AND RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print("""
Run this script to identify which of the three problems is causing
MusicHal's poor performance:

1. Token Collapse - If all gestures map to token 16, the system is perceptually blind
2. Frame Availability - If audio_frames lack MIDI data, interval extraction fails
3. Pattern Mismatch - If rhythm pattern densities don't match, queries return 0

Next steps based on findings:
- Token collapse → Retrain vocabulary with more diverse data
- Missing MIDI in frames → Fix training pipeline to preserve pitch data
- Density mismatch → Adjust RhythmOracle query density calculation

Check the output above for specific issues flagged with ❌ or ⚠️
""")
