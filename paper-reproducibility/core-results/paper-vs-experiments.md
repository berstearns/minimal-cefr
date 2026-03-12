# Paper Tables vs Experiment Results

Side-by-side comparison of values in the published paper tables and the values
extracted from experiment data in this repository.

**Note**: The paper includes perplexity-based features (native, native+general AL,
all models) from a separate pipeline not stored in these experiment folders.
Only the tf-idf rows can be compared. AC2 values differ because the paper uses
a weighted Gwet's AC2 implementation; values here use a simplified version.

---

## Table: f1-summary-per-scenario (Macro F1)

### Paper values

| Model & Setup | CELVA-SP (zs) | KUPA-KEYS (zs) | EFC-test (zs) | Avg (zs) | CELVA-SP (90-10) | KUPA-KEYS (90-10) | EFC-test (90-10) | Avg (90-10) |
|---|---|---|---|---|---|---|---|---|
| **Statistical Models** | | | | | | | | |
| Majority Class (oracle) | 0.086 | 0.121 | 0.107 | 0.105 | 0.086 | 0.180 | 0.128 | 0.131 |
| LR native perpl. | 0.157 | 0.228 | 0.538 | 0.308 | 0.286 | 0.255 | 0.513 | 0.351 |
| LR native + general AL perpl. | 0.153 | 0.197 | 0.657 | 0.336 | 0.286 | 0.250 | 0.640 | 0.392 |
| LR all models perpl. | 0.156 | 0.285 | 0.499 | 0.313 | 0.346 | 0.305 | 0.520 | 0.390 |
| LR tf-idf | 0.193 | 0.192 | 0.731 | 0.372 | 0.283 | 0.278 | 0.635 | 0.399 |
| XGBoost native perpl. | 0.169 | 0.238 | 0.741 | 0.383 | 0.314 | 0.315 | 0.575 | 0.401 |
| XGBoost native + general AL perpl. | 0.173 | 0.228 | 0.736 | 0.379 | 0.314 | 0.310 | 0.570 | 0.398 |
| **XGBoost all models perpl.** | 0.177 | 0.269 | 0.541 | 0.329 | **0.345** | **0.320** | **0.673** | **0.446** |
| **XGBoost tf-idf** | **0.197** | **0.188** | **0.785** | **0.390** | 0.294 | 0.282 | 0.638 | 0.405 |
| **Text Classifiers** | | | | | | | | |
| BERT | 0.128 | 0.253 | 0.380 | 0.254 | - | - | - | - |
| **Prompt Models** | | | | | | | | |
| Gemma 2B AES1 | 0.063 | 0.174 | 0.022 | 0.086 | - | - | - | - |
| Gemma 2B AES2 | 0.063 | 0.199 | 0.022 | 0.095 | - | - | - | - |
| Gemma 7B AES1 | 0.171 | 0.160 | 0.149 | 0.160 | - | - | - | - |
| Gemma 7B AES2 | 0.118 | 0.104 | 0.149 | 0.124 | - | - | - | - |
| LLaMA 3 8B AES1 | 0.183 | 0.050 | 0.198 | 0.144 | - | - | - | - |
| LLaMA 3 8B AES2 | 0.142 | 0.093 | 0.198 | 0.144 | - | - | - | - |
| Mistral 7B AES1 | 0.178 | 0.047 | 0.193 | 0.139 | - | - | - | - |
| Mistral 7B AES2 | 0.132 | 0.068 | 0.193 | 0.131 | - | - | - | - |

### Experiment data (tf-idf only — best per model across all tf-idf configs)

| Model & Setup | CELVA-SP (zs) | KUPA-KEYS (zs) | EFC-test (zs) | Avg (zs) | CELVA-SP (90-10) | KUPA-KEYS (90-10) | EFC-test (90-10) | Avg (90-10) |
|---|---|---|---|---|---|---|---|---|
| LR tf-idf (best) | 0.156 | 0.300 | 0.922 | 0.459 | 0.350 | 0.207 | n/a | 0.279 |
| LR tf-idf (ng1-2, 5k) | 0.148 | 0.234 | 0.906 | 0.429 | 0.332 | 0.157 | n/a | 0.244 |
| LR tf-idf-grp (best) | 0.154 | 0.300 | 0.912 | 0.455 | n/a | n/a | n/a | n/a |
| XGB tf-idf (best) | 0.217 | 0.194 | 0.901 | 0.437 | 0.329 | 0.123 | n/a | 0.226 |
| XGB tf-idf (ng1-2, 5k) | 0.204 | 0.181 | 0.886 | 0.424 | 0.303 | 0.123 | n/a | 0.213 |
| XGB tf-idf-grp (best) | 0.217 | 0.194 | 0.901 | 0.437 | n/a | n/a | n/a | n/a |
| **Prompt Models** | | | | | | | | |
| Gemma 2B AES2 | 0.055 | 0.183 | n/a | n/a | - | - | - | - |
| Gemma 7B AES2 | 0.170 | 0.203 | n/a | n/a | - | - | - | - |
| LLaMA 3 8B AES2 | 0.183 | 0.040 | n/a | n/a | - | - | - | - |
| Mistral 7B AES2 | 0.179 | 0.032 | n/a | n/a | - | - | - | - |

---

## Table: Numerical CEFR Grading — CELVA-SP

### Paper values (zero-shot only, tf-idf rows)

| Model | Features | N | RMSE | Within1 | Spearman ρ | AC2 |
|---|---|---|---|---|---|---|
| LR tf-idf | 1742 | 1.255 | 0.761 | 0.490 | 0.828 |
| XGBoost tf-idf | 1742 | 1.200 | 0.791 | 0.474 | 0.838 |
| LLaMA 3 8B AES1 | 1742 | 1.016 | 0.877 | 0.539 | 0.910 |
| Mistral 7B AES2 | 1742 | **0.955** | **0.903** | **0.579** | **0.917** |

### Experiment data (zero-shot, best tf-idf config)

| Model | Features | N | RMSE | Within1 | Spearman |
|---|---|---|---|---|---|
| LR tfidf 10000 ng1-2 | zero-shot-2 | 1742 | 1.422 | 0.445 | 0.449 |
| LR tfidf 5000 ng1-1 | zero-shot-2 | 1742 | 1.383 | 0.474 | 0.461 |
| XGB tfidf 5000 ng2-3 | zero-shot-2 | 1742 | 1.089 | 0.632 | 0.309 |
| XGB tfidf 10000 ng1-2 | zero-shot-2 | 1742 | 1.089 | 0.613 | 0.374 |
| LLaMA 3 8B AES2 | prompting | 1742 | 1.016 | 0.877 | 0.330 |
| Mistral 7B AES2 | prompting | 1742 | 1.037 | 0.871 | 0.413 |

### Experiment data (90-10, best tf-idf config)

| Model | Features | N | RMSE | Within1 | Spearman |
|---|---|---|---|---|---|
| LR tfidf 5000 ng1-2 | 90-10 | 175 | 0.745 | 0.829 | 0.753 |
| LR tfidf 10000 ng1-2 | 90-10 | 175 | 0.751 | 0.811 | 0.763 |
| XGB tfidf 10000 ng1-2 | 90-10 | 175 | 0.797 | 0.817 | 0.653 |
| XGB tfidf 1000 ng1-2 | 90-10 | 175 | 0.804 | 0.811 | 0.649 |

---

## Table: Numerical CEFR Grading — KUPA-KEYS

### Paper values (zero-shot only, tf-idf and prompt rows)

| Model | Features | N | RMSE | Within1 | Spearman ρ | AC2 |
|---|---|---|---|---|---|---|
| LR tf-idf | 1006 | 0.838 | 0.953 | 0.497 | 0.914 |
| XGBoost tf-idf | 1006 | 0.982 | 0.880 | 0.319 | 0.869 |
| XGB native+gen AL | **1006** | **0.707** | **0.984** | **0.503** | **0.906** |
| GPT-4o-mini AES2 | 1006 | 2.085 | 0.841 | **0.624** | 0.779 |

### Experiment data (zero-shot, best tf-idf config)

| Model | Features | N | RMSE | Within1 | Spearman |
|---|---|---|---|---|---|
| LR tfidf 10000 ng1-2 | zero-shot-2 | 1006 | 0.659 | 0.887 | 0.268 |
| LR tfidf 5000 ng1-2 | zero-shot-2 | 1006 | 0.669 | 0.883 | 0.252 |
| XGB tfidf-grp 1000x5 ng1-3 | zero-shot-2 | 1006 | 0.947 | 0.695 | 0.102 |
| Gemma 7B AES2 | prompting | 1006 | 1.140 | 0.873 | 0.288 |

### Experiment data (90-10, best tf-idf config)

| Model | Features | N | RMSE | Within1 | Spearman |
|---|---|---|---|---|---|
| LR tfidf 1000 ng1-2 | 90-10 | 101 | 0.833 | 0.772 | 0.356 |
| LR tfidf 100 ng1-2 | 90-10 | 101 | 0.857 | 0.772 | 0.285 |
| XGB tfidf 5000 ng1-2 | 90-10 | 101 | 1.082 | 0.624 | 0.328 |

---

## Table: Numerical CEFR Grading — EFCAMDAT-test (in-domain)

### Experiment data (zero-shot, best configs)

| Model | Features | N | RMSE | Within1 | Spearman |
|---|---|---|---|---|---|
| LR tfidf-grp 2000x5 ng1-2 | zero-shot-2 | 20002 | 0.238 | 0.988 | 0.920 |
| LR tfidf 10000 ng1-2 | zero-shot-2 | 20002 | 0.248 | 0.987 | 0.918 |
| XGB tfidf 5000 ng1-1 | zero-shot-2 | 20002 | 0.250 | 0.987 | 0.920 |
| XGB tfidf-grp 2000x5 ng1-2 | zero-shot-2 | 20002 | 0.243 | 0.987 | 0.921 |
| XGB tfidf-grp 1000x5 ng1-2 | zero-shot-2 | 20002 | 0.253 | 0.986 | 0.920 |
