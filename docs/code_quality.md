# Code Quality Report

Generated: 2025-10-18

## Summary

This report contains all code quality issues found in the `src/` directory (excluding `extract_perplexity_features.py`).

---

## FLAKE8 ERRORS (Linting & PEP 8)

### High Priority Issues

#### 1. Bare `except` clauses (E722) - 3 occurrences
**Risk**: Can catch system exits and keyboard interrupts, making debugging difficult.

- `predict.py:442:9` - do not use bare 'except'
- `train_classifiers.py:280:13` - do not use bare 'except'
- `train_classifiers_with_ho_multifeat.py:899:9` - do not use bare 'except'

#### 2. Ambiguous variable names (E741) - 12 occurrences
**Risk**: Poor readability, easy to confuse with numbers.

- `predict.py:507:38` - ambiguous variable name 'l'
- `predict.py:529:42` - ambiguous variable name 'l'
- `predict.py:637:46` - ambiguous variable name 'l'
- `predict.py:659:46` - ambiguous variable name 'l'
- `predict.py:683:50` - ambiguous variable name 'l'
- `predict.py:698:50` - ambiguous variable name 'l'
- `predict.py:915:38` - ambiguous variable name 'l'
- `predict.py:937:42` - ambiguous variable name 'l'
- `predict.py:1034:46` - ambiguous variable name 'l'
- `predict.py:1055:42` - ambiguous variable name 'l'
- `predict.py:1079:46` - ambiguous variable name 'l'
- `predict.py:1094:46` - ambiguous variable name 'l'

#### 3. Indentation issues (E128) - 27 occurrences
**Risk**: Inconsistent formatting, harder to read.

**predict.py:**
- `562:38` - continuation line under-indented
- `962:38` - continuation line under-indented

**train_classifiers_with_ho_multifeat.py:**
- Lines: 958, 970, 971, 972, 1024, 1028, 1030, 1034, 1036, 1040, 1042, 1046, 1047, 1048, 1049, 1050, 1051, 1060, 1061, 1062, 1066, 1071, 1073, 1075, 1077, 1081, 1085, 1087

#### 4. Unused variables (F841) - 5 occurrences
**Risk**: Dead code, potential bugs.

- `predict.py:319:5` - local variable 'label_to_idx' assigned but never used
- `predict.py:595:13` - local variable 'class_to_idx' assigned but never used
- `predict.py:993:13` - local variable 'class_to_idx' assigned but never used
- `train_classifiers_with_ho.py:304:9` - local variable 'e' assigned but never used
- `train_classifiers_with_ho_multifeat.py:956:13` - local variable 'metric_name' assigned but never used

#### 5. Unused imports (F401) - 16 occurrences
**Risk**: Clutters namespace, increases load time.

- `config.py:13:1` - 'typing.List' imported but unused
- `extract_features.py:321:1` - 'typing.Optional' imported but unused
- `extract_features.py:323:1` - 'numpy as np' imported but unused
- `extract_features.py:327:1` - 'src.train_tfidf_groupby.GroupedTfidfVectorizer' imported but unused
- `predict.py:15:1` - 'typing.Dict' imported but unused
- `predict.py:24:1` - 'src.train_tfidf_groupby.GroupedTfidfVectorizer' imported but unused
- `report.py:12:1` - 'typing.Tuple' imported but unused
- `train_classifiers_with_ho.py:16:1` - 'sklearn.metrics.log_loss' imported but unused
- `train_classifiers_with_ho.py:32` - Multiple unused imports from src.config (ExperimentConfig, ClassifierConfig, DataConfig, OutputConfig)
- `train_classifiers_with_ho_multifeat.py:15:1` - 'pickle' imported but unused
- `train_classifiers_with_ho_multifeat.py:38` - Multiple unused imports from src.config
- `train_tfidf.py:12:1` - 'typing.Optional' imported but unused
- `train_tfidf_groupby.py:14:1` - 'typing.Optional' imported but unused

#### 6. F-strings without placeholders (F541) - 20 occurrences
**Risk**: Inefficient, should use regular strings.

- `config.py:30:30`
- `extract_features.py:466:23`
- `pipeline.py:57:15, 201:23, 229:35, 234:31`
- `predict.py:718:27, 719:23, 721:27, 723:23, 793:27, 1114:27, 1115:23, 1117:27, 1118:19, 1392:31`
- `train_classifiers.py:245:15`
- `train_classifiers_with_ho.py:378:15, 389:15, 405:19, 426:19, 522:15, 738:23`
- `train_tfidf_groupby.py:244:19`

### Medium Priority Issues

#### 7. Cyclomatic complexity warnings (C901) - 35 occurrences
**Risk**: Functions too complex, hard to test and maintain.

- `extract_features.py:330` - 'extract_features_for_file' (complexity: 20)
- `extract_features.py:471` - 'extract_all_from_source' (complexity: 12)
- `extract_features.py:628` - 'args_to_config' (complexity: 15)
- `extract_features.py:699` - 'main' (complexity: 15)
- `pipeline.py:24` - 'run_pipeline' (complexity: 48) **⚠️ CRITICAL**
- `pipeline.py:1455` - 'args_to_config' (complexity: 22)
- `predict.py:402` - 'predict_on_features' (complexity: 40) **⚠️ CRITICAL**
- `predict.py:728` - 'predict_all_feature_sets' (complexity: 15)
- `predict.py:813` - 'predict_with_text_pipeline' (complexity: 33)
- `predict.py:1257` - 'args_to_config' (complexity: 15)
- `predict.py:1330` - 'main' (complexity: 19)
- `report.py:41` - 'parse_evaluation_report' (complexity: 12)
- `report.py:353` - 'generate_summary_report' (complexity: 16)
- `train_classifiers.py:118` - 'load_features_and_labels' (complexity: 12)
- `train_classifiers.py:201` - 'train_classifier' (complexity: 15)
- `train_classifiers.py:353` - 'train_all_classifiers' (complexity: 15)
- `train_classifiers.py:616` - 'args_to_config' (complexity: 23)
- `train_classifiers.py:718` - 'main' (complexity: 14)
- `train_classifiers_with_ho.py:51` - 'load_features_and_labels' (complexity: 12)
- `train_classifiers_with_ho.py:325` - 'train_classifier_with_ho' (complexity: 17)
- `train_classifiers_with_ho.py:529` - 'batch_train_classifiers_with_ho' (complexity: 14)
- `train_classifiers_with_ho.py:618` - 'main' (complexity: 13)
- `train_classifiers_with_ho_multifeat.py:222` - 'compute_metric' (complexity: 11)
- `train_classifiers_with_ho_multifeat.py:284` - 'load_features_and_labels_from_dir' (complexity: 11)
- `train_classifiers_with_ho_multifeat.py:455` - 'stage1_screen_features' (complexity: 22)
- `train_classifiers_with_ho_multifeat.py:621` - 'stage2_optimize_top_features' (complexity: 14)
- `train_tfidf.py:19` - 'train_tfidf' (complexity: 12)
- `train_tfidf.py:209` - 'args_to_config' (complexity: 17)
- `train_tfidf_groupby.py:161` - 'train_grouped_tfidf' (complexity: 16)
- `train_tfidf_groupby.py:413` - 'args_to_config' (complexity: 17)

---

## FORMATTING ISSUES (BLACK & ISORT)

These can be auto-fixed by running: `cd utils && ./code_quality.sh ../src fix`

**Files requiring black formatting (10):**
- config.py
- extract_features.py
- pipeline.py
- predict.py
- report.py
- train_classifiers.py
- train_classifiers_with_ho.py
- train_classifiers_with_ho_multifeat.py
- train_tfidf.py
- train_tfidf_groupby.py

**Files requiring isort import sorting (10):**
- Same as above

---

## RECOMMENDED FIX ORDER

1. **Auto-fix formatting** (safe, no functionality risk)
   - Run: `cd utils && ./code_quality.sh ../src fix`

2. **Remove unused imports** (F401) - Low risk

3. **Fix f-string placeholders** (F541) - Low risk

4. **Rename ambiguous variables** (E741: 'l' → 'label' or 'level')

5. **Fix bare except clauses** (E722) - Add specific exception types

6. **Remove unused variables** (F841) - or use them if needed

7. **Fix indentation** (E128) - May be auto-fixed by black

8. **Refactor complex functions** (C901) - Requires careful refactoring

---

## FILES EXCLUDED FROM THIS REPORT

- `extract_perplexity_features.py` (per user request - will be handled manually)
- `__pycache__/` directory
- `__init__.py` (no errors found)

---

## NEXT STEPS

Please authorize fixes for each category:
1. ✓ Auto-fix formatting and imports? (BLACK + ISORT)
2. ✓ Remove unused imports? (F401)
3. ✓ Convert f-strings to regular strings? (F541)
4. ✓ Rename variable 'l' to 'label'? (E741)
5. ✓ Fix bare except clauses? (E722)
6. ✓ Remove unused variables? (F841)
7. ✓ Fix indentation issues? (E128)
