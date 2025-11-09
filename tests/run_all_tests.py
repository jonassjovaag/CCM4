"""
Master Test Runner
==================
Executes all test files in sequence and reports results.

Tests run in order:
1. test_model_inspection.py - Model structure diagnostic
2. test_note_extraction.py - Verify bug fix with real model
3. test_synthetic_events.py - Mock data injection
4. test_end_to_end_flow.py - Complete flow verification
5. test_viewport_translation.py - Chord name field diagnostic

Exit code 0 if all pass, 1 if any fail.

IMPORTANT: Uses CCM3 virtual environment (CCM3/bin/python)
"""

import subprocess
import sys
from pathlib import Path


def get_venv_python():
    """Get the Python executable from CCM3 virtual environment."""
    project_root = Path(__file__).parent.parent
    venv_python = project_root / "CCM3" / "bin" / "python"
    
    if venv_python.exists():
        return str(venv_python)
    else:
        print(f"⚠️  Warning: CCM3 venv not found at {venv_python}")
        print(f"   Falling back to system Python: {sys.executable}")
        return sys.executable


def run_test(test_name, test_file, python_exe):
    """Run a single test file and return result."""
    
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {test_name}")
    print(f"{'=' * 80}\n")
    
    try:
        result = subprocess.run(
            [python_exe, test_file],
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n❌ FAILED to run {test_name}: {e}")
        return False


def main():
    """Run all tests in sequence."""
    
    print("=" * 80)
    print("MUSICHAL 9000 TEST SUITE")
    print("=" * 80)
    
    # Get CCM3 venv Python
    python_exe = get_venv_python()
    print(f"\nUsing Python: {python_exe}")
    
    print("\nRunning all tests in sequence...")
    print("Each test is ISOLATED - no modifications to live system")
    
    tests_dir = Path(__file__).parent
    
    # Define test sequence
    tests = [
        ("Test 1: Model Inspection", tests_dir / "test_model_inspection.py"),
        ("Test 2: Note Extraction", tests_dir / "test_note_extraction.py"),
        ("Test 3: Synthetic Events", tests_dir / "test_synthetic_events.py"),
        ("Test 4: End-to-End Flow", tests_dir / "test_end_to_end_flow.py"),
        ("Test 5: Viewport Translation", tests_dir / "test_viewport_translation.py"),
    ]
    
    results = {}
    
    # Run each test
    for test_name, test_file in tests:
        if not test_file.exists():
            print(f"\n❌ SKIP: {test_name} - file not found: {test_file}")
            results[test_name] = False
            continue
        
        success = run_test(test_name, test_file, python_exe)
        results[test_name] = success
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUITE SUMMARY")
    print(f"{'=' * 80}\n")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n{'=' * 80}")
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"{'=' * 80}\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉\n")
        print("The bug fix is verified:")
        print("  - Dictionary lookup working (frame_id in audio_frames)")
        print("  - MIDI field name fallbacks working")
        print("  - End-to-end flow functioning correctly")
        print("\nSafe to test live with MusicHal_9000.py")
        return 0
    else:
        print(f"⚠️  {failed} TEST(S) FAILED\n")
        print("Review failed tests above before live testing")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
