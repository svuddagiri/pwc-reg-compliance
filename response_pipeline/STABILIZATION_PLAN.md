# 🎯 PWC Regulatory System Stabilization Plan

## Overview
This plan addresses the core issues affecting the regulatory compliance system:
1. Accuracy of results/missing results  
2. Citation mismatches
3. Junk text in responses
4. Performance issues
5. **NEW**: Exact clause/article text without paraphrasing (like GPT)

**Core Principle**: Fix issues without breaking existing functionality

## Phase 1: Baseline & Benchmarking 📊

### Step 1.1: GPT Benchmark Integration
- [ ] **User Action**: Export GPT results to `/tests/gpt_benchmark_andrea_questions.txt`
- [ ] Create benchmark comparison tool (`pipeline/benchmark_comparison.py`)
- [ ] Implement semantic similarity scoring vs GPT results
- [ ] Create accuracy metrics dashboard

### Step 1.2: Current State Assessment  
- [ ] Run comprehensive test on all 4 Andrea questions
- [ ] Document current performance metrics
- [ ] Create regression test suite
- [ ] Backup current working state

### Step 1.3: Issue Categorization
- [ ] **Accuracy Issues**: Missing jurisdictions, incomplete responses
- [ ] **Citation Issues**: Wrong article numbers, format mismatches  
- [ ] **Quality Issues**: Junk text, poor formatting
- [ ] **Performance Issues**: >10s response times
- [ ] **Exact Text Issues**: Paraphrasing instead of verbatim quotes

## Phase 2: Targeted Fixes 🔧

### Fix 1: Exact Text Extraction (HIGH PRIORITY)
**Timeline**: 2-3 hours
**Scope**: Show clauses/articles AS-IS without paraphrasing, exactly like GPT

**Problem**: Our system paraphrases regulatory text instead of showing exact quotes
**GPT Approach**: Shows verbatim text with exact article numbers and document details

**Actions**:
- [ ] **Analyze GPT's exact text format** from benchmark file
- [ ] **Identify source of paraphrasing** (prompt instructions vs context processing)
- [ ] **Add "verbatim quote" instructions** to prompts
- [ ] **Test chunk content** - verify it contains exact regulatory text (not summaries)
- [ ] **Implement exact text extraction**:
  ```
  Expected Format (matching GPT):
  | Topic | Article/Section | Exact Quoted Text | Source |
  | Consent Definition | Art. 2(f) | "Consent of the owner of the personal data: Any expression of free, unequivocal, informed and specific wish..." | Costa Rica Regulation 37554-JP |
  ```
- [ ] **Ensure article numbers match source documents** exactly
- [ ] **Add document details** (regulation names, article numbers, page references)
- [ ] **Test against GPT benchmark** - exact text must match character-for-character where possible

### Fix 2: Prompt Template Audit
**Timeline**: 1-2 hours
**Scope**: Review and clean prompts without changing core logic

```bash
# Files to review:
- /prompts/prompts.json (system prompts)
- /src/services/response_generator.py (dynamic prompts)
```

**Actions**:
- [ ] Remove conflicting instructions in prompts
- [ ] **Add explicit "DO NOT PARAPHRASE" instructions**
- [ ] **Add "Show exact regulatory text first" requirements**
- [ ] Standardize output format requirements
- [ ] Test each prompt change against benchmark
- [ ] Rollback any change that reduces accuracy

### Fix 3: Citation Extraction Stabilization  
**Timeline**: 2-3 hours
**Scope**: Ensure citations match GPT's exact format

**Actions**:
- [ ] Compare our citation format vs GPT benchmark
- [ ] **Ensure article numbers come from original documents** (not LLM generated)
- [ ] Test regex patterns against actual chunk metadata
- [ ] Implement fallback citation methods
- [ ] **Validate article numbers against source documents**
- [ ] **Add document hierarchy information** (regulation name, section, subsection)

### Fix 4: Response Quality Control
**Timeline**: 1-2 hours  
**Scope**: Remove junk text and ensure consistent formatting

**Actions**:
- [ ] Add output validation to response generator
- [ ] Clean up template formatting instructions
- [ ] **Implement structured response format** (like GPT's table format)
- [ ] Test response structure consistency
- [ ] Implement response post-processing if needed

### Fix 5: Performance Optimization
**Timeline**: 2-3 hours
**Scope**: Target specific bottlenecks without changing accuracy

**Actions**:
- [ ] Profile each pipeline stage timing
- [ ] Implement caching for concept expansion  
- [ ] Optimize prompt token usage
- [ ] Test DEMO_MODE effectiveness

## Phase 3: Validation & Testing ✅

### Step 3.1: Benchmark Testing
- [ ] Run all 4 Andrea questions after each fix
- [ ] Compare results vs GPT benchmark (>80% similarity target)
- [ ] **Verify exact text matches GPT's verbatim quotes**
- [ ] **Check article numbers match GPT's citations exactly**
- [ ] Ensure no regression in working questions
- [ ] Document improvement metrics

### Step 3.2: Exact Text Validation
- [ ] **Compare our "exact quotes" vs GPT's exact quotes**
- [ ] **Validate article numbers against original PDF documents**
- [ ] **Check document references match GPT's format**
- [ ] **Ensure no paraphrasing in regulatory text sections**

### Step 3.3: Regression Prevention
- [ ] Create automated test suite
- [ ] Implement change validation pipeline
- [ ] Set up monitoring for accuracy degradation
- [ ] Document "do not change" critical components

## Implementation Strategy 🚀

### Week 1: Immediate Fixes
**Day 1**: Exact Text Extraction & Benchmark Setup (HIGHEST PRIORITY)
**Day 2**: Prompt Template Audit (Focus on "no paraphrase" instructions)
**Day 3-4**: Citation & Quality Fixes  
**Day 5**: Performance Optimization & Testing

### Success Metrics
- [ ] **Accuracy**: >80% semantic similarity to GPT benchmark
- [ ] **Exact Text**: Verbatim quotes match GPT's quotes
- [ ] **Citations**: Article numbers match GPT's citations exactly
- [ ] **Document Details**: Regulation names, sections match GPT format
- [ ] **Quality**: No junk text, consistent formatting
- [ ] **Performance**: <10s average response time
- [ ] **Stability**: No regression in previously working features

### Key Focus Areas (In Priority Order)
1. **Exact Text Extraction** - Show verbatim regulatory language
2. **Article Number Accuracy** - Match source documents exactly  
3. **Document References** - Include regulation names, sections
4. **Response Structure** - Table format like GPT
5. **Performance** - Maintain speed while improving accuracy

## Tools & Scripts 🛠️

### New Tools to Create
1. **Benchmark Comparison Tool** (`pipeline/benchmark_comparison.py`)
   - Semantic similarity scoring
   - **Exact text comparison** (character-by-character where applicable)
   - **Citation format comparison** (article numbers, document names)
   - Jurisdiction coverage analysis
   - Performance metrics tracking

2. **Exact Text Validator** (`pipeline/validate_exact_text.py`)
   - **Compare our quotes vs GPT quotes**
   - **Validate article numbers against source PDFs**
   - **Check for paraphrasing vs verbatim text**
   - **Document reference accuracy**

3. **Regression Test Suite** (`pipeline/regression_test.py`)
   - Quick validation of all 4 questions
   - Pass/fail against benchmark
   - **Exact text validation**
   - Performance monitoring
   - Automated rollback triggers

### Existing Tools to Use
- `pipeline/test_andrea_quick.py` (already fixed)
- Checkpoint restore capability (already available)
- DEMO_MODE for performance testing

## Decision Points 🤔

### When to Proceed vs Rollback
**Proceed if**:
- Accuracy improves or stays same vs benchmark
- **Exact text quotations improve** (less paraphrasing)
- **Article numbers become more accurate**
- No new errors introduced
- Performance doesn't degrade significantly

**Rollback if**:
- Any accuracy reduction vs GPT benchmark
- **More paraphrasing than before**
- **Article numbers become less accurate**
- New errors in previously working questions
- Significant performance degradation

### Change Approval Criteria
1. **Benchmark Test**: Must pass semantic similarity test
2. **Exact Text Test**: Must show improvement in verbatim quotes
3. **Citation Test**: Article numbers must be more accurate
4. **Regression Test**: All 4 questions must complete successfully  
5. **Performance Test**: Response time <15s (target <10s)
6. **User Validation**: User confirms improvement vs current state

## Specific Technical Tasks 🔧

### Task 1: Investigate Paraphrasing Source
- [ ] Check if chunks contain exact regulatory text or summaries
- [ ] Review prompt instructions that might cause paraphrasing
- [ ] Test if LLM is paraphrasing despite having exact text
- [ ] Compare context sent to LLM vs GPT's source material

### Task 2: Implement Exact Text Instructions
- [ ] Add to prompts: "NEVER paraphrase regulatory text"
- [ ] Add: "Show exact quoted text with quotation marks"
- [ ] Add: "Include precise article numbers as they appear in source"
- [ ] Add: "Use document names exactly as they appear in metadata"

### Task 3: Validate Source Material
- [ ] Check PDF documents for exact article numbers GPT uses
- [ ] Verify our chunks contain the same exact text
- [ ] Test if article numbers in metadata match document structure
- [ ] Ensure regulation names in metadata are accurate

## Next Steps 📋

1. **User**: Export GPT benchmark results to text file
2. **Development**: Create exact text comparison tool
3. **Investigation**: Analyze why we paraphrase vs show exact text
4. **Implementation**: Begin with exact text extraction (highest priority)
5. **Validation**: Test each change against GPT benchmark

## Emergency Rollback Plan 🆘

If any change breaks the system:
1. Restore from checkpoint: `response_pipeline_checkpoint_20250706_221318`
2. Document what broke
3. Reassess approach
4. Test smaller incremental changes

---

**Remember**: The goal is to match GPT's exact text presentation - verbatim quotes with accurate article numbers and document details. Every change must be validated against the GPT benchmark.