#!/usr/bin/env python3
"""
Verify token_frequencies sum for harmonic and percussive vocabularies
"""

import joblib

print("=" * 80)
print("Harmonic vocabulary:")
harmonic = joblib.load('input_audio/General_idea_harmonic_vocab.joblib')
token_freq = harmonic['token_frequencies']
total = sum(token_freq.values())
print(f"Total samples: {total:,}")
print(f"Number of unique tokens: {len(token_freq)}")
print(f"Top 5 tokens:")
sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:5]
for token_id, count in sorted_tokens:
    print(f"  Token {token_id}: {count:,} ({100*count/total:.2f}%)")

print("\n" + "=" * 80)
print("Percussive vocabulary:")
percussive = joblib.load('input_audio/General_idea_percussive_vocab.joblib')
token_freq_perc = percussive['token_frequencies']
total_perc = sum(token_freq_perc.values())
print(f"Total samples: {total_perc:,}")
print(f"Number of unique tokens: {len(token_freq_perc)}")
print(f"Top 5 tokens:")
sorted_tokens = sorted(token_freq_perc.items(), key=lambda x: x[1], reverse=True)[:5]
for token_id, count in sorted_tokens:
    print(f"  Token {token_id}: {count:,} ({100*count/total_perc:.2f}%)")

print("\n" + "=" * 80)
print("Combined:")
print(f"Harmonic samples: {total:,}")
print(f"Percussive samples: {total_perc:,}")
print(f"Total: {total + total_perc:,}")
print(f"  → If training extracted from BOTH harmonic + percussive: {total + total_perc:,} samples")
print(f"  → Matches 49,865? {'✅ YES' if abs((total + total_perc) - 49865) < 100 else '❌ NO'}")
