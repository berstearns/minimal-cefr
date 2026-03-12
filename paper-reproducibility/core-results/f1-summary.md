# Macro F1 Summary by Scenario

Matching paper table: `f1-summary-per-scenario.tex`

## Statistical Models

| Model | Features | CELVA (zs) | KUPA (zs) | EFC-test (zs) | Avg (zs) | CELVA (90-10) | KUPA (90-10) | EFC-test (90-10) | Avg (90-10) |
|-------|----------|------------|-----------|---------------|----------|---------------|--------------|------------------|-------------|
| logistic | hash:252cd532 | 0.143 | 0.232 | 0.905 | 0.427 | n/a | n/a | n/a | n/a |
| logistic | hash:336a6205 | 0.123 | 0.295 | 0.818 | 0.412 | n/a | n/a | n/a | n/a |
| logistic | tfidf 100 ng1-2 | n/a | n/a | n/a | n/a | 0.261 | 0.207 | n/a | 0.234 |
| logistic | tfidf 1000 ng1-2 | 0.123 | 0.295 | 0.818 | 0.412 | 0.317 | 0.163 | n/a | 0.240 |
| logistic | tfidf 10000 ng1-2 | 0.147 | 0.236 | 0.922 | 0.435 | 0.350 | 0.158 | n/a | 0.254 |
| logistic | tfidf 2000 ng1-2 | 0.137 | 0.228 | 0.861 | 0.409 | n/a | n/a | n/a | n/a |
| logistic | tfidf 500 ng1-2 | 0.117 | 0.225 | 0.777 | 0.373 | n/a | n/a | n/a | n/a |
| logistic | tfidf 5000 ng1-1 | 0.156 | 0.229 | 0.915 | 0.433 | n/a | n/a | n/a | n/a |
| logistic | tfidf 5000 ng1-2 | 0.148 | 0.234 | 0.906 | 0.429 | 0.332 | 0.157 | n/a | 0.244 |
| logistic | tfidf 5000 ng1-3 | 0.145 | 0.231 | 0.898 | 0.425 | n/a | n/a | n/a | n/a |
| logistic | tfidf 5000 ng2-2 | 0.137 | 0.233 | 0.825 | 0.398 | n/a | n/a | n/a | n/a |
| logistic | tfidf 5000 ng2-3 | 0.135 | 0.230 | 0.809 | 0.391 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 1000x5=5000 ng1-1 | 0.154 | 0.229 | 0.897 | 0.427 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 1000x5=5000 ng1-2 | 0.150 | 0.296 | 0.887 | 0.444 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 1000x5=5000 ng1-3 | 0.151 | 0.293 | 0.886 | 0.443 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 1000x5=5000 ng2-2 | 0.152 | 0.300 | 0.832 | 0.428 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 1000x5=5000 ng2-3 | 0.152 | 0.243 | 0.817 | 0.404 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 100x5=500 ng1-2 | 0.115 | 0.213 | 0.709 | 0.346 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 2000x5=10000 ng1-2 | 0.149 | 0.234 | 0.912 | 0.432 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 200x5=1000 ng1-2 | 0.131 | 0.292 | 0.782 | 0.402 | n/a | n/a | n/a | n/a |
| logistic | tfidf-grp 500x5=2500 ng1-2 | 0.145 | 0.291 | 0.851 | 0.429 | n/a | n/a | n/a | n/a |
| xgboost | hash:252cd532 | 0.204 | 0.172 | 0.886 | 0.421 | n/a | n/a | n/a | n/a |
| xgboost | hash:336a6205 | 0.193 | 0.174 | 0.851 | 0.406 | n/a | n/a | n/a | n/a |
| xgboost | hash:341c9ba5 | 0.202 | 0.177 | 0.846 | 0.408 | n/a | n/a | n/a | n/a |
| xgboost | hash:7b271c2d | 0.211 | 0.177 | 0.730 | 0.373 | n/a | n/a | n/a | n/a |
| xgboost | hash:84cbc90c | 0.199 | 0.177 | 0.783 | 0.386 | n/a | n/a | n/a | n/a |
| xgboost | hash:c2b5a010 | 0.202 | 0.169 | 0.882 | 0.418 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 100 ng1-2 | n/a | n/a | n/a | n/a | 0.273 | 0.117 | n/a | 0.195 |
| xgboost | tfidf 1000 ng1-2 | 0.193 | 0.174 | 0.851 | 0.406 | 0.319 | 0.107 | n/a | 0.213 |
| xgboost | tfidf 10000 ng1-2 | 0.211 | 0.138 | 0.893 | 0.414 | 0.329 | 0.114 | n/a | 0.222 |
| xgboost | tfidf 2000 ng1-2 | 0.208 | 0.178 | 0.863 | 0.416 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 500 ng1-2 | 0.205 | 0.179 | 0.820 | 0.401 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 5000 ng1-1 | 0.188 | 0.177 | 0.896 | 0.420 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 5000 ng1-2 | 0.204 | 0.181 | 0.886 | 0.424 | 0.303 | 0.123 | n/a | 0.213 |
| xgboost | tfidf 5000 ng1-3 | 0.198 | 0.177 | 0.881 | 0.419 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 5000 ng2-2 | 0.210 | 0.139 | 0.775 | 0.375 | n/a | n/a | n/a | n/a |
| xgboost | tfidf 5000 ng2-3 | 0.216 | 0.146 | 0.762 | 0.375 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 1000x5=5000 ng1-1 | 0.198 | 0.193 | 0.891 | 0.427 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 1000x5=5000 ng1-2 | 0.185 | 0.176 | 0.891 | 0.417 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 1000x5=5000 ng1-3 | 0.190 | 0.187 | 0.887 | 0.421 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 1000x5=5000 ng2-2 | 0.202 | 0.150 | 0.814 | 0.389 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 1000x5=5000 ng2-3 | 0.217 | 0.152 | 0.805 | 0.391 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 100x5=500 ng1-2 | 0.199 | 0.177 | 0.783 | 0.386 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 2000x5=10000 ng1-2 | 0.208 | 0.185 | 0.901 | 0.431 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 200x5=1000 ng1-2 | 0.191 | 0.194 | 0.837 | 0.407 | n/a | n/a | n/a | n/a |
| xgboost | tfidf-grp 500x5=2500 ng1-2 | 0.196 | 0.185 | 0.873 | 0.418 | n/a | n/a | n/a | n/a |

## Prompt Models

| Model | Prompt | CELVA-SP | KUPA-KEYS | EFCAMDAT-test |
|-------|--------|----------|-----------|---------------|
| Gemma 2B | AES2 | 0.055 | 0.183 | n/a |
| Gemma 7B | AES2 | 0.170 | 0.203 | n/a |
| LLaMA 3 8B | AES2 | 0.183 | 0.040 | n/a |
| Mistral 7B | AES2 | 0.179 | 0.032 | n/a |
