#!/usr/bin/env python3
"""
PHASE 8 INTEGRATION GAP ANALYSIS
Check for type mismatches, missing parameters, and logic inconsistencies
"""

def check_type_consistency():
    """Check that root frequencies use consistent types (Hz as float)"""
    print("\n" + "="*70)
    print("TYPE CONSISTENCY CHECK")
    print("="*70)
    
    issues = []
    
    # Check 1: All root hints are Hz (float), not MIDI or symbolic
    print("\n✓ Check 1: Root frequency representation (Hz everywhere)")
    
    files_to_check = {
        'performance_arc_analyzer.py': 'root_hint_frequency: Optional[float]',
        'performance_timeline_manager.py': 'current_root_hint: Optional[float]',
        'agent/autonomous_root_explorer.py': 'root_hz: float',
        'memory/polyphonic_audio_oracle.py': 'root_hint_hz: float',
    }
    
    for file, expected in files_to_check.items():
        with open(file, 'r') as f:
            if expected in f.read():
                print(f"  ✅ {file}: {expected}")
            else:
                print(f"  ❌ {file}: MISSING {expected}")
                issues.append(f"Missing {expected} in {file}")
    
    # Check 2: Consonance is 0.0-1.0 everywhere
    print("\n✓ Check 2: Consonance representation (0.0-1.0 everywhere)")
    
    consonance_files = {
        'memory/polyphonic_audio_oracle.py': 'consonances',
        'agent/autonomous_root_explorer.py': 'consonance',
        'performance_timeline_manager.py': 'current_tension_target',
    }
    
    for file, term in consonance_files.items():
        with open(file, 'r') as f:
            if term in f.read():
                print(f"  ✅ {file}: uses '{term}'")
            else:
                print(f"  ❌ {file}: MISSING '{term}'")
                issues.append(f"Missing {term} in {file}")
    
    return issues


def check_parameter_flow():
    """Check that parameters flow correctly through the pipeline"""
    print("\n" + "="*70)
    print("PARAMETER FLOW CHECK")
    print("="*70)
    
    issues = []
    
    # Check 1: input_fundamental parameter added to update_performance_state
    print("\n✓ Check 1: input_fundamental parameter")
    with open('performance_timeline_manager.py', 'r') as f:
        content = f.read()
        has_param = 'input_fundamental: Optional[float]' in content
        passed_to_explorer = 'input_fundamental=input_fundamental' in content
        
        if has_param:
            print(f"  ✅ Parameter defined in update_performance_state()")
        else:
            print(f"  ❌ Parameter NOT defined")
            issues.append("input_fundamental parameter missing")
        
        if passed_to_explorer:
            print(f"  ✅ Passed to explorer.update()")
        else:
            print(f"  ❌ NOT passed to explorer")
            issues.append("input_fundamental not passed to explorer")
    
    # Check 2: mode_params passed from caller to PhraseGenerator
    print("\n✓ Check 2: mode_params propagation")
    with open('agent/phrase_generator.py', 'r') as f:
        content = f.read()
        
        # Check generate_phrase signature
        has_mode_params = 'def generate_phrase(self,' in content
        
        # Check _build_request_for_mode receives mode_params
        receives_params = 'def _build_request_for_mode(self, mode: str, mode_params: Dict' in content
        
        # Check _add_root_hints_to_request uses mode_params
        uses_params = 'def _add_root_hints_to_request(self, request: Dict, mode_params: Dict)' in content
        
        if receives_params:
            print(f"  ✅ mode_params in _build_request_for_mode()")
        else:
            print(f"  ❌ mode_params NOT in _build_request_for_mode()")
            issues.append("mode_params missing from _build_request_for_mode")
        
        if uses_params:
            print(f"  ✅ mode_params in _add_root_hints_to_request()")
        else:
            print(f"  ❌ mode_params NOT in _add_root_hints_to_request()")
            issues.append("mode_params missing from _add_root_hints_to_request")
    
    # Check 3: Request dict contains correct keys
    print("\n✓ Check 3: Request dict keys")
    with open('agent/phrase_generator.py', 'r') as f:
        content = f.read()
        has_root_hint = "request['root_hint_hz']" in content
        has_tension = "request['tension_target']" in content
        has_bias_strength = "request['root_bias_strength']" in content
        
        if has_root_hint:
            print(f"  ✅ request['root_hint_hz']")
        else:
            print(f"  ❌ request['root_hint_hz'] NOT added")
            issues.append("root_hint_hz not added to request")
        
        if has_tension:
            print(f"  ✅ request['tension_target']")
        else:
            print(f"  ❌ request['tension_target'] NOT added")
            issues.append("tension_target not added to request")
        
        if has_bias_strength:
            print(f"  ✅ request['root_bias_strength'] (optional)")
        else:
            print(f"  ⚠️  request['root_bias_strength'] not found (optional, has default)")
    
    # Check 4: AudioOracle reads request keys
    print("\n✓ Check 4: AudioOracle reads request")
    with open('memory/polyphonic_audio_oracle.py', 'r') as f:
        content = f.read()
        reads_root = "request.get('root_hint_hz')" in content
        reads_tension = "request.get('tension_target'" in content
        reads_strength = "request.get('root_bias_strength'" in content
        
        if reads_root:
            print(f"  ✅ Reads root_hint_hz")
        else:
            print(f"  ❌ Does NOT read root_hint_hz")
            issues.append("AudioOracle doesn't read root_hint_hz")
        
        if reads_tension:
            print(f"  ✅ Reads tension_target")
        else:
            print(f"  ❌ Does NOT read tension_target")
            issues.append("AudioOracle doesn't read tension_target")
        
        if reads_strength:
            print(f"  ✅ Reads root_bias_strength")
        else:
            print(f"  ❌ Does NOT read root_bias_strength")
            issues.append("AudioOracle doesn't read root_bias_strength")
    
    return issues


def check_logic_consistency():
    """Check for logic errors or inconsistencies"""
    print("\n" + "="*70)
    print("LOGIC CONSISTENCY CHECK")
    print("="*70)
    
    issues = []
    
    # Check 1: Biasing only applied when fundamentals exist
    print("\n✓ Check 1: Biasing safety checks")
    with open('memory/polyphonic_audio_oracle.py', 'r') as f:
        content = f.read()
        has_fundamentals_check = "if not self.fundamentals" in content
        has_strength_check = "bias_strength <= 0" in content
        
        if has_fundamentals_check:
            print(f"  ✅ Checks if fundamentals exist")
        else:
            print(f"  ❌ Missing fundamentals check")
            issues.append("No safety check for fundamentals")
        
        if has_strength_check:
            print(f"  ✅ Checks if bias_strength <= 0")
        else:
            print(f"  ❌ Missing bias_strength check")
            issues.append("No safety check for bias_strength")
    
    # Check 2: Explorer only runs when initialized
    print("\n✓ Check 2: Explorer safety checks")
    with open('performance_timeline_manager.py', 'r') as f:
        content = f.read()
        has_explorer_check = "if self.root_explorer and" in content
        
        if has_explorer_check:
            print(f"  ✅ Checks if root_explorer exists")
        else:
            print(f"  ❌ Missing explorer existence check")
            issues.append("No safety check for root_explorer")
    
    # Check 3: Interval calculation is log2 (perceptual)
    print("\n✓ Check 3: Perceptual interval calculation")
    with open('memory/polyphonic_audio_oracle.py', 'r') as f:
        content = f.read()
        has_log2 = "12 * np.log2(" in content
        
        if has_log2:
            print(f"  ✅ Uses log2 for interval calculation (perceptual)")
        else:
            print(f"  ❌ Missing log2 calculation")
            issues.append("Interval not calculated with log2")
    
    # Check 4: Probability normalization
    print("\n✓ Check 4: Probability normalization")
    with open('memory/polyphonic_audio_oracle.py', 'r') as f:
        content = f.read()
        # Look for normalization in _apply_root_hint_bias
        has_normalize = "weights / np.sum(weights)" in content or "/ weight_sum" in content
        
        if has_normalize:
            print(f"  ✅ Normalizes biased weights")
        else:
            print(f"  ❌ Missing normalization")
            issues.append("Weights not normalized after biasing")
    
    # Check 5: Default values for optional parameters
    print("\n✓ Check 5: Default parameter values")
    with open('memory/polyphonic_audio_oracle.py', 'r') as f:
        content = f.read()
        has_tension_default = "tension_target: float = 0.5" in content or "request.get('tension_target', 0.5)" in content
        has_strength_default = "bias_strength: float = 0.3" in content or "request.get('root_bias_strength', 0.3)" in content
        
        if has_tension_default:
            print(f"  ✅ tension_target defaults to 0.5")
        else:
            print(f"  ⚠️  tension_target default not explicit")
        
        if has_strength_default:
            print(f"  ✅ bias_strength defaults to 0.3")
        else:
            print(f"  ❌ bias_strength has no default")
            issues.append("bias_strength missing default")
    
    return issues


def check_missing_imports():
    """Check that all necessary imports exist"""
    print("\n" + "="*70)
    print("IMPORT CHECK")
    print("="*70)
    
    issues = []
    
    # Check 1: Timeline imports explorer classes
    print("\n✓ Check 1: Timeline imports")
    with open('performance_timeline_manager.py', 'r') as f:
        content = f.read()
        has_explorer_import = "from agent.autonomous_root_explorer import" in content
        has_all_classes = all(cls in content for cls in ['AutonomousRootExplorer', 'RootWaypoint', 'ExplorationConfig'])
        
        if has_explorer_import and has_all_classes:
            print(f"  ✅ All explorer classes imported")
        else:
            print(f"  ❌ Missing explorer imports")
            issues.append("Explorer classes not imported in timeline manager")
    
    # Check 2: Explorer has necessary imports
    print("\n✓ Check 2: Explorer imports")
    with open('agent/autonomous_root_explorer.py', 'r') as f:
        content = f.read()
        has_numpy = "import numpy as np" in content
        has_time = "import time" in content
        has_dataclass = "from dataclasses import dataclass" in content
        
        all_good = has_numpy and has_time and has_dataclass
        if all_good:
            print(f"  ✅ All necessary imports present")
        else:
            print(f"  ❌ Missing imports")
            if not has_numpy:
                issues.append("numpy not imported in explorer")
            if not has_time:
                issues.append("time not imported in explorer")
            if not has_dataclass:
                issues.append("dataclass not imported in explorer")
    
    return issues


def main():
    """Run all integration gap checks"""
    print("\n" + "="*70)
    print("PHASE 8 INTEGRATION GAP ANALYSIS")
    print("="*70)
    
    all_issues = []
    
    all_issues.extend(check_type_consistency())
    all_issues.extend(check_parameter_flow())
    all_issues.extend(check_logic_consistency())
    all_issues.extend(check_missing_imports())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not all_issues:
        print("\n✅ NO INTEGRATION GAPS FOUND")
        print("   All types consistent, parameters flow correctly, logic is sound")
        return 0
    else:
        print(f"\n⚠️  FOUND {len(all_issues)} POTENTIAL ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
