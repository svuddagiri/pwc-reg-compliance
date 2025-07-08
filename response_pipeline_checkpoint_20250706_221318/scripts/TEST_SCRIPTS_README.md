# Failed Questions Test Suite

This directory contains comprehensive test scripts for verifying the fixes implemented for previously failing questions (Q2, Q6, Q9, Q11, Q12).

## Scripts Overview

### 1. `test_failed_questions.py`
Main test script that runs all failed questions through the API and collects metrics.

**Features:**
- Tests all 5 previously failed questions
- Measures response times
- Checks for expected keywords
- Calculates success rates
- Saves results to JSON file

**Usage:**
```bash
python scripts/test_failed_questions.py
```

**Output:**
- Console display with color-coded results
- JSON file: `test_results_YYYYMMDD_HHMMSS.json`

### 2. `test_individual_fixes.py`
Unit-level tests for individual fixes to verify they work correctly.

**Tests:**
- Term normalizer (Q2 - Costa Rica)
- Query parsing (Q9 - no truncation)
- Fallback handlers (Q6, Q11)
- Keyword extraction (Q12)

**Usage:**
```bash
python scripts/test_individual_fixes.py
```

### 3. `compare_test_results.py`
Compares test results between runs to show improvements.

**Features:**
- Side-by-side comparison
- Success rate changes
- Performance metrics
- Keyword coverage improvements

**Usage:**
```bash
# Compare two specific files
python scripts/compare_test_results.py test_results_before.json test_results_after.json

# Or auto-detect latest two files
python scripts/compare_test_results.py
```

### 4. `generate_improvement_report.py`
Generates a comprehensive markdown report documenting all fixes and results.

**Features:**
- Executive summary
- Detailed fix documentation
- Test results integration
- Implementation details
- Recommendations

**Usage:**
```bash
python scripts/generate_improvement_report.py
```

**Output:**
- Console display
- Markdown file: `improvement_report_YYYYMMDD_HHMMSS.md`

## Test Workflow

1. **Ensure server is running:**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

2. **Run individual fix verification:**
   ```bash
   python scripts/test_individual_fixes.py
   ```

3. **Run full test suite:**
   ```bash
   python scripts/test_failed_questions.py
   ```

4. **Compare results (if you have previous run):**
   ```bash
   python scripts/compare_test_results.py
   ```

5. **Generate final report:**
   ```bash
   python scripts/generate_improvement_report.py
   ```

## Expected Results

After implementing all fixes, you should see:

| Question | Expected Improvement |
|----------|---------------------|
| Q2 | Costa Rica now appears in results |
| Q6 | GDPR penalties and fines described |
| Q9 | Full query processed without truncation |
| Q11 | Cross-border transfer mechanisms explained |
| Q12 | Periodic audit requirements found |

## Success Criteria

- **Success Rate:** Should be 80-100% (4-5 out of 5 questions passing)
- **Keyword Coverage:** At least 50% of expected keywords found
- **Response Time:** Under 10 seconds per query (ideally under 5s)

## Troubleshooting

If tests fail:

1. **Check server is running:** Look for connection errors
2. **Review individual fixes:** Run `test_individual_fixes.py`
3. **Check API logs:** Look for errors in server console
4. **Verify database:** Ensure tracking tables are accessible

## Files Generated

- `test_results_*.json` - Raw test results
- `improvement_report_*.md` - Formatted improvement report
- Console output with color-coded results