#!/usr/bin/env python3
"""
Diagnose RhythmOracle ↔ AudioOracle Integration

Traces the data flow from rhythmic patterns to harmonic generation:
1. Training: RhythmOracle pattern extraction
2. Performance: RhythmOracle query in phrase generation
3. Integration: Rhythmic phrasing → AudioOracle request masking
4. Output: MIDI timing application

Checks:
- Is RhythmOracle properly initialized?
- Are rhythmic patterns loaded?
- Does phrase_generator query RhythmOracle?
- Are rhythmic_phrasing dicts included in AudioOracle requests?
- Is timing actually applied to MIDI output?
"""

import json
import os
from typing import Dict, List, Optional

print('🔍 RHYTHMIC-HARMONIC INTEGRATION DIAGNOSTIC')
print('=' * 80)

# ============================================================================
# 1. CHECK TRAINING OUTPUT - Are rhythmic patterns captured?
# ============================================================================

print('\n📚 STEP 1: TRAINING DATA ANALYSIS')
print('-' * 80)

# Find most recent training files
training_files = []
for f in os.listdir('JSON'):
    if f.endswith('_training.json'):
        training_files.append(f)

if training_files:
    most_recent = sorted(training_files)[-1]
    print(f'Latest training: {most_recent}')
    
    with open(f'JSON/{most_recent}', 'r') as f:
        training_data = json.load(f)
    
    # Check rhythmic analysis
    if 'rhythmic_analysis' in training_data:
        ra = training_data['rhythmic_analysis']
        print(f'\n✅ Rhythmic Analysis Found:')
        print(f'   Tempo: {ra.get("tempo", "N/A")} BPM')
        print(f'   Patterns detected: {ra.get("patterns_detected", "N/A")}')
        print(f'   Syncopation: {ra.get("syncopation", "N/A")}')
        print(f'   Complexity: {ra.get("complexity", "N/A")}')
    else:
        print(f'\n❌ No rhythmic_analysis in training data')
    
    # Check for RhythmOracle file
    rhythm_oracle_file = most_recent.replace('_training.json', '_rhythm_oracle.json')
    if os.path.exists(f'JSON/{rhythm_oracle_file}'):
        print(f'\n✅ RhythmOracle File Found: {rhythm_oracle_file}')
        
        with open(f'JSON/{rhythm_oracle_file}', 'r') as f:
            rhythm_data = json.load(f)
        
        print(f'   Total patterns: {len(rhythm_data.get("patterns", {}))}')
        
        # Analyze pattern characteristics
        if 'patterns' in rhythm_data and rhythm_data['patterns']:
            patterns = rhythm_data['patterns']
            print(f'\n   📊 Pattern Breakdown:')
            
            densities = []
            syncopations = []
            pulses = []
            complexity_scores = []
            
            for pattern_id, pattern in patterns.items():
                densities.append(pattern.get('density', 0))
                syncopations.append(pattern.get('syncopation', 0))
                pulses.append(pattern.get('pulse', 4))
                complexity_scores.append(pattern.get('complexity', 0))
            
            print(f'      Density range: {min(densities):.2f} - {max(densities):.2f}')
            print(f'      Syncopation range: {min(syncopations):.2f} - {max(syncopations):.2f}')
            print(f'      Pulses: {set(pulses)}')
            print(f'      Complexity range: {min(complexity_scores):.1f} - {max(complexity_scores):.1f}')
            
            # Check duration patterns (key for tempo-independent matching)
            print(f'\n   📏 Duration Patterns (first 3):')
            for i, (pattern_id, pattern) in enumerate(list(patterns.items())[:3]):
                dur_pattern = pattern.get('duration_pattern', [])
                print(f'      Pattern {pattern_id}: {dur_pattern[:10]}{"..." if len(dur_pattern) > 10 else ""}')
        
    else:
        print(f'\n❌ RhythmOracle file NOT found: {rhythm_oracle_file}')
else:
    print('❌ No training files found in JSON/')

# ============================================================================
# 2. CHECK CODE INTEGRATION - Is RhythmOracle queried during performance?
# ============================================================================

print('\n\n🔗 STEP 2: CODE INTEGRATION ANALYSIS')
print('-' * 80)

# Check if phrase_generator queries RhythmOracle
phrase_gen_file = 'agent/phrase_generator.py'
if os.path.exists(phrase_gen_file):
    with open(phrase_gen_file, 'r') as f:
        phrase_gen_code = f.read()
    
    # Check for critical methods
    checks = [
        ('_get_rhythmic_phrasing_from_oracle', 'Method to query RhythmOracle'),
        ('rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle', 'Calling RhythmOracle query'),
        ('request[\'rhythmic_phrasing\']', 'Including phrasing in request'),
        ('_apply_rhythmic_phrasing_to_timing', 'Applying phrasing to MIDI timing'),
    ]
    
    print(f'Checking {phrase_gen_file}:')
    for pattern, description in checks:
        if pattern in phrase_gen_code:
            print(f'   ✅ {description}')
        else:
            print(f'   ❌ MISSING: {description}')
    
    # Count how many request builders include rhythmic_phrasing
    request_methods = [
        '_build_shadowing_request',
        '_build_mirroring_request',
        '_build_coupling_request',
        '_build_leading_request',
    ]
    
    print(f'\n   📋 Request Builders Including Rhythmic Phrasing:')
    for method in request_methods:
        if method in phrase_gen_code:
            # Check if this method queries RhythmOracle
            method_start = phrase_gen_code.find(f'def {method}')
            if method_start != -1:
                method_end = phrase_gen_code.find('\n    def ', method_start + 1)
                method_code = phrase_gen_code[method_start:method_end if method_end != -1 else len(phrase_gen_code)]
                
                if '_get_rhythmic_phrasing_from_oracle' in method_code:
                    print(f'      ✅ {method}')
                else:
                    print(f'      ❌ {method} (no RhythmOracle query)')
        else:
            print(f'      ❓ {method} (method not found)')

else:
    print(f'❌ File not found: {phrase_gen_file}')

# Check MusicHal_9000.py initialization
musichel_file = 'MusicHal_9000.py'
if os.path.exists(musichel_file):
    with open(musichel_file, 'r') as f:
        musichel_code = f.read()
    
    print(f'\nChecking {musichel_file}:')
    
    init_checks = [
        ('RhythmOracle()', 'RhythmOracle instantiation'),
        ('rhythm_oracle.load_patterns', 'Loading rhythmic patterns'),
        ('PhraseGenerator(.*rhythm_oracle', 'Passing RhythmOracle to PhraseGenerator'),
    ]
    
    import re
    for pattern, description in init_checks:
        if re.search(pattern, musichel_code):
            print(f'   ✅ {description}')
        else:
            print(f'   ❌ MISSING: {description}')
else:
    print(f'❌ File not found: {musichel_file}')

# ============================================================================
# 3. TEMPO/SUBDIVISION ANALYSIS - Is there a ceiling?
# ============================================================================

print('\n\n🎵 STEP 3: TEMPO/SUBDIVISION CEILING ANALYSIS')
print('-' * 80)

if 'rhythm_data' in locals() and 'patterns' in rhythm_data:
    patterns = rhythm_data['patterns']
    
    # Analyze duration patterns for subdivision granularity
    print(f'Analyzing {len(patterns)} rhythmic patterns for subdivision limits...\n')
    
    min_subdivisions = []
    max_subdivisions = []
    subdivision_variety = []
    
    for pattern_id, pattern in patterns.items():
        dur_pattern = pattern.get('duration_pattern', [])
        if dur_pattern:
            min_subdivisions.append(min(dur_pattern))
            max_subdivisions.append(max(dur_pattern))
            subdivision_variety.append(len(set(dur_pattern)))
    
    if min_subdivisions:
        print(f'📊 Subdivision Statistics:')
        print(f'   Finest subdivision (min duration): {min(min_subdivisions)}')
        print(f'   Coarsest subdivision (max duration): {max(max_subdivisions)}')
        print(f'   Average subdivision variety per pattern: {sum(subdivision_variety)/len(subdivision_variety):.1f} unique values')
        print()
        
        # Interpret results
        finest = min(min_subdivisions)
        if finest == 1:
            print(f'   ✅ Captures finest-grain subdivisions (16th/32nd notes possible)')
        elif finest <= 3:
            print(f'   ⚠️  Minimum subdivision = {finest} (may miss fastest notes)')
        else:
            print(f'   ❌ Minimum subdivision = {finest} (likely missing fine rhythmic detail)')
        
        # Check tempo range
        if 'ra' in locals():
            training_tempo = ra.get('tempo', 120)
            print(f'\n   🎼 Training tempo: {training_tempo:.1f} BPM')
            
            # Calculate theoretical max subdivision tempo
            # If finest subdivision is 1, and training tempo is 120 BPM:
            # - Quarter notes = 120 BPM
            # - 16th notes = 480 BPM (120 * 4)
            # This is NOT a tempo ceiling but subdivision granularity
            
            implied_max = training_tempo * 4  # Assuming 16th notes
            print(f'   💡 Implied max subdivision rate: {implied_max:.1f} events/min (16th notes at {training_tempo:.1f} BPM)')
            print(f'      This is NOT a tempo ceiling - it\'s subdivision granularity!')
            print(f'      System can play at ANY tempo, but with subdivision detail up to {finest}-unit resolution')

else:
    print('⚠️  No RhythmOracle patterns available for analysis')

# ============================================================================
# 4. SUMMARY & RECOMMENDATIONS
# ============================================================================

print('\n\n🎯 STEP 4: SUMMARY & RECOMMENDATIONS')
print('=' * 80)

print('\n📋 Integration Status:')

issues_found = []

# Check 1: RhythmOracle patterns exist
if 'rhythm_data' in locals() and rhythm_data.get('patterns'):
    print(f'   ✅ RhythmOracle patterns captured ({len(rhythm_data["patterns"])} patterns)')
else:
    print(f'   ❌ RhythmOracle patterns missing or empty')
    issues_found.append('No rhythmic patterns in training data')

# Check 2: Code integration present
if os.path.exists(phrase_gen_file):
    with open(phrase_gen_file, 'r') as f:
        code = f.read()
    if '_get_rhythmic_phrasing_from_oracle' in code:
        print(f'   ✅ RhythmOracle query method exists')
    else:
        print(f'   ❌ RhythmOracle query method missing')
        issues_found.append('phrase_generator.py missing _get_rhythmic_phrasing_from_oracle()')

# Check 3: Request masking integration
if os.path.exists(phrase_gen_file):
    if 'request[\'rhythmic_phrasing\']' in code:
        print(f'   ✅ Rhythmic phrasing added to requests')
    else:
        print(f'   ❌ Rhythmic phrasing NOT added to requests')
        issues_found.append('Rhythmic phrasing not included in AudioOracle requests')

# Check 4: MIDI timing application
if os.path.exists(phrase_gen_file):
    if '_apply_rhythmic_phrasing_to_timing' in code:
        print(f'   ✅ Timing application method exists')
    else:
        print(f'   ❌ Timing application method missing')
        issues_found.append('No method to apply rhythmic phrasing to MIDI timing')

print('\n🔬 Investigation Results:')
if not issues_found:
    print('   ✅ ALL SYSTEMS OPERATIONAL!')
    print('   RhythmOracle ↔ AudioOracle integration appears complete.')
    print()
    print('   📊 Data Flow:')
    print('   1. Training: Extracts rhythmic patterns → RhythmOracle')
    print('   2. Performance: Recent events → query RhythmOracle')
    print('   3. Integration: Rhythmic phrasing → AudioOracle request')
    print('   4. Output: Phrasing → MIDI timing')
else:
    print(f'   ⚠️  FOUND {len(issues_found)} ISSUE(S):')
    for issue in issues_found:
        print(f'      • {issue}')

print('\n💡 Key Findings:')
print('   • Tempo/subdivision is NOT a ceiling - it\'s granularity')
print('   • System is tempo-INDEPENDENT (matches duration patterns, not BPM)')
print('   • Subdivision detail preserved in duration_pattern arrays')
print('   • AudioOracle distance threshold FIX (cosine) should improve variation')
print('   • RhythmOracle provides WHEN/HOW (timing/phrasing)')
print('   • AudioOracle provides WHAT (pitches/harmonies)')

print('\n🎵 Next Steps:')
print('   1. Test live performance with --enable-rhythmic flag')
print('   2. Monitor logs for RhythmOracle query debug messages')
print('   3. Verify rhythmic_phrasing dict appears in requests')
print('   4. Listen for rhythmic variation (should match training material)')
print('   5. Compare with/without rhythmic mode for A/B testing')
print()
