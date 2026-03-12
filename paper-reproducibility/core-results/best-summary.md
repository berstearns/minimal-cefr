# Best Results Summary

## Best Macro F1 per Dataset

| Dataset | Best Model | Features | Experiment | M-F1 | Acc | Adj.Acc | RMSE | Within1 |
|---------|-----------|----------|------------|------|-----|---------|------|---------|
| CELVA-SP | logistic | tfidf 10000 ng1-2 | 90-10 | **0.350** | 0.486 | 0.914 | 0.751 | 0.811 |
| KUPA-KEYS | logistic | tfidf-grp 1000x5=5000 ng2-2 | zero-shot-2 | **0.300** | 0.486 | 0.941 | 0.725 | 0.838 |
| EFCAMDAT-test | logistic | tfidf 10000 ng1-2 | zero-shot-2 | **0.922** | 0.964 | 0.992 | 0.248 | 0.987 |

## Best RMSE per Dataset

| Dataset | Best Model | Features | Experiment | RMSE | Within1 | Spearman | AC2 |
|---------|-----------|----------|------------|------|---------|----------|-----|
| CELVA-SP | logistic | tfidf 5000 ng1-2 | 90-10 | **0.745** | 0.829 | 0.753 | 0.619 |
| KUPA-KEYS | logistic | tfidf 10000 ng1-2 | zero-shot-2 | **0.659** | 0.887 | 0.268 | 0.189 |
| EFCAMDAT-test | logistic | tfidf-grp 2000x5=10000 ng1-2 | zero-shot-2 | **0.238** | 0.988 | 0.920 | 0.966 |

## Comparison with Paper Table Values

Paper `f1-summary-per-scenario.tex` reference values:

| Row | CELVA-SP (zs) | KUPA-KEYS (zs) | EFC-test (zs) |
|-----|---------------|----------------|---------------|
| XGBoost tf-idf (paper) | 0.197 | 0.188 | 0.785 |
| Logistic tf-idf (paper) | 0.193 | 0.192 | 0.731 |
  - xgboost tfidf on CELVA-SP (ours): 0.217 (tfidf-grp 1000x5=5000 ng2-3, zero-shot-2)
  - xgboost tfidf on KUPA-KEYS (ours): 0.194 (tfidf-grp 200x5=1000 ng1-2, zero-shot-2)
  - xgboost tfidf on EFCAMDAT-test (ours): 0.901 (tfidf-grp 2000x5=10000 ng1-2, zero-shot-2)
  - logistic tfidf on CELVA-SP (ours): 0.156 (tfidf 5000 ng1-1, zero-shot-2)
  - logistic tfidf on KUPA-KEYS (ours): 0.300 (tfidf-grp 1000x5=5000 ng2-2, zero-shot-2)
  - logistic tfidf on EFCAMDAT-test (ours): 0.922 (tfidf 10000 ng1-2, zero-shot-2)
