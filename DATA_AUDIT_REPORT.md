# Data Integrity Audit Report

**Directory:** `JSON`

**Date:** <map object at 0x000001FA81F64DC0>

```
================================================================================
DATA INTEGRITY AUDIT REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total files audited:  183
Valid files:          181 (98.9%)
Invalid files:        2 (1.1%)

FILE TYPES
--------------------------------------------------------------------------------
  audio_oracle                  3 files
  correlation                   7 files
  harmonic_transitions          2 files
  rhythm_oracle                26 files
  training_results            110 files
  unknown                      35 files

INVALID FILES (ERRORS)
--------------------------------------------------------------------------------

File: JSON\Curious_child_091125_1912_training_harmonic_transitions.json
  Size: 0 bytes
  ERROR: JSON parse error: Expecting value: line 1 column 1 (char 0)

File: JSON\performancedata.json
  Size: 17,253 bytes
  ERROR: JSON parse error: Extra data: line 2 column 1 (char 188)

FILES WITH WARNINGS (DATA QUALITY ISSUES)
--------------------------------------------------------------------------------
Total: 36 files


File: debug_test.json
  WARNING: Unknown file type: unknown

File: grab-a-hold_enhanced.json
  WARNING: Unknown file type: unknown

File: grab-a-hold_enhanced_hybrid_stats.json
  WARNING: Unknown file type: unknown

File: hybrid_short_test.json
  WARNING: Unknown file type: unknown

File: hybrid_short_test_hybrid_stats.json
  WARNING: Unknown file type: unknown

File: hybrid_test.json
  WARNING: Unknown file type: unknown

File: hybrid_test_stats.json
  WARNING: Unknown file type: unknown

File: simple_test.json
  WARNING: Unknown file type: unknown

File: simple_test_stats.json
  WARNING: Unknown file type: unknown

File: test_enhanced.json
  WARNING: Unknown file type: unknown

... and 26 more files with warnings

STATISTICS SUMMARY
--------------------------------------------------------------------------------
Total audio frames:   4,000
Total oracle states:  167,177
Total patterns:       1,281,222

================================================================================
```
