#!/usr/bin/env python3
"""
Check existing quantizer diversity
"""

import joblib
import sys
from collections import Counter

def check_quantizer_diversity(quantizer_file):
    """Check diversity of existing trained quantizer"""
    
    print(f"🔍 Analyzing: {quantizer_file}")
    print("=" * 50)
    
    try:
        # Load quantizer
        data = joblib.load(quantizer_file)
        
        if isinstance(data, dict):
            print("📊 Quantizer Statistics:")
            print(f"   Vocabulary size: {data.get('vocabulary_size', 'Unknown')}")
            print(f"   Feature dimension: {data.get('feature_dim', 'Unknown')}")
            print(f"   Uses L2 normalization: {data.get('use_l2_norm', 'Unknown')}")
            
            if 'token_frequencies' in data:
                token_freqs = data['token_frequencies']
                print(f"\n🎯 Token Usage:")
                print(f"   Active tokens: {len(token_freqs)}/64")
                print(f"   Total assignments: {sum(token_freqs.values())}")
                
                # Calculate entropy
                total = sum(token_freqs.values())
                if total > 0:
                    import numpy as np
                    probs = np.array([count / total for count in token_freqs.values()])
                    entropy = -np.sum(probs * np.log2(probs + 1e-9))
                    print(f"   Entropy: {entropy:.2f} bits")
                
                # Top tokens
                sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
                print(f"\n   Top 10 most used tokens:")
                for i, (token, count) in enumerate(sorted_tokens[:10]):
                    pct = count / total * 100
                    print(f"      {i+1}. Token {token}: {count} uses ({pct:.1f}%)")
                
                # Check if any tokens dominate
                max_usage = max(token_freqs.values())
                if max_usage / total > 0.2:
                    print(f"\n⚠️  Token {sorted_tokens[0][0]} dominates with {max_usage/total*100:.1f}% usage!")
                
                if len(token_freqs) < 32:
                    print(f"\n⚠️  Low diversity: Only {len(token_freqs)}/64 tokens used")
                else:
                    print(f"\n✅ Good diversity: {len(token_freqs)}/64 tokens used")
            
        else:
            print("❌ Unexpected quantizer format")
            
    except Exception as e:
        print(f"❌ Error loading quantizer: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_quantizer.py <quantizer_file>")
        sys.exit(1)
    
    check_quantizer_diversity(sys.argv[1])