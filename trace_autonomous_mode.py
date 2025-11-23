#!/usr/bin/env python3
"""
Trace autonomous_mode changes by monkey-patching PhraseGenerator.
"""

import time
import sys
import traceback

# Monkey-patch PhraseGenerator to trace autonomous_mode changes
original_setattr = None

def trace_setattr(self, name, value):
    """Trace all autonomous_mode changes"""
    if name == 'autonomous_mode':
        stack = ''.join(traceback.format_stack()[:-1])
        timestamp = time.time()
        print(f"\n{'='*80}")
        print(f"🔍 TRACE: autonomous_mode changed to {value} at {timestamp:.3f}")
        print(f"{'='*80}")
        print(stack)
        print(f"{'='*80}\n")
    
    # Call original __setattr__
    original_setattr(self, name, value)

def install_trace():
    """Install the trace monkey-patch"""
    from agent.phrase_generator import PhraseGenerator
    global original_setattr
    
    # Save original __setattr__
    original_setattr = PhraseGenerator.__setattr__
    
    # Replace with tracing version
    PhraseGenerator.__setattr__ = trace_setattr
    
    print("🔍 Autonomous mode tracer installed!")
    print("   Will print stack trace every time autonomous_mode changes")

if __name__ == '__main__':
    # Install trace before importing MusicHal
    install_trace()
    
    # Now run MusicHal with original arguments
    import sys
    sys.argv = [
        'MusicHal_9000.py',
        '--enable-somax',
        '--enable-meld', 
        '--performance-duration', '1'  # 1 minute test
    ]
    
    # Import and run MusicHal
    from scripts.performance.MusicHal_9000 import main
    main()
