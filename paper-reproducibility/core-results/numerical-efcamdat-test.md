# Numerical CEFR Grading: EFCAMDAT-test

Matching paper tables: `efcamdat-test-numerical-both-scenarios.tex`

## Statistical Models

| Model | Features | Scenario | N | RMSE | Within1 | Spearman | AC2 | M-F1 | Acc |
|-------|----------|----------|---|------|---------|----------|-----|------|-----|
| logistic | hash:252cd532 | zero-shot | 20002 | 0.261 | 0.985 | 0.917 | 0.960 | 0.905 | 0.958 |
| logistic | hash:336a6205 | zero-shot | 20002 | 0.340 | 0.972 | 0.909 | 0.935 | 0.818 | 0.915 |
| xgboost | hash:252cd532 | zero-shot | 20002 | 0.258 | 0.985 | 0.920 | 0.958 | 0.886 | 0.949 |
| xgboost | hash:336a6205 | zero-shot | 20002 | 0.287 | 0.981 | 0.916 | 0.947 | 0.851 | 0.936 |
| xgboost | hash:341c9ba5 | zero-shot | 20002 | 0.291 | 0.980 | 0.916 | 0.945 | 0.846 | 0.934 |
| xgboost | hash:7b271c2d | zero-shot | 20002 | 0.384 | 0.966 | 0.899 | 0.902 | 0.730 | 0.871 |
| xgboost | hash:84cbc90c | zero-shot | 20002 | 0.335 | 0.974 | 0.909 | 0.926 | 0.783 | 0.902 |
| xgboost | hash:c2b5a010 | zero-shot | 20002 | 0.260 | 0.985 | 0.920 | 0.957 | 0.882 | 0.948 |
| logistic | tfidf 1000 ng1-2 | zero-shot-2 | 20002 | 0.340 | 0.972 | 0.909 | 0.935 | 0.818 | 0.915 |
| logistic | tfidf 10000 ng1-2 | zero-shot-2 | 20002 | 0.248 | 0.987 | 0.918 | 0.964 | 0.922 | 0.964 |
| logistic | tfidf 2000 ng1-2 | zero-shot-2 | 20002 | 0.296 | 0.980 | 0.915 | 0.949 | 0.861 | 0.939 |
| logistic | tfidf 500 ng1-2 | zero-shot-2 | 20002 | 0.396 | 0.961 | 0.900 | 0.910 | 0.777 | 0.882 |
| logistic | tfidf 5000 ng1-1 | zero-shot-2 | 20002 | 0.265 | 0.983 | 0.915 | 0.958 | 0.915 | 0.956 |
| logistic | tfidf 5000 ng1-2 | zero-shot-2 | 20002 | 0.261 | 0.985 | 0.917 | 0.960 | 0.905 | 0.958 |
| logistic | tfidf 5000 ng1-3 | zero-shot-2 | 20002 | 0.264 | 0.984 | 0.917 | 0.960 | 0.898 | 0.956 |
| logistic | tfidf 5000 ng2-2 | zero-shot-2 | 20002 | 0.342 | 0.971 | 0.906 | 0.931 | 0.825 | 0.926 |
| logistic | tfidf 5000 ng2-3 | zero-shot-2 | 20002 | 0.356 | 0.969 | 0.904 | 0.924 | 0.809 | 0.918 |
| logistic | tfidf-grp 1000x5=5000 ng1-1 | zero-shot-2 | 20002 | 0.258 | 0.986 | 0.918 | 0.960 | 0.897 | 0.950 |
| logistic | tfidf-grp 1000x5=5000 ng1-2 | zero-shot-2 | 20002 | 0.264 | 0.984 | 0.917 | 0.958 | 0.887 | 0.947 |
| logistic | tfidf-grp 1000x5=5000 ng1-3 | zero-shot-2 | 20002 | 0.265 | 0.985 | 0.917 | 0.957 | 0.886 | 0.945 |
| logistic | tfidf-grp 1000x5=5000 ng2-2 | zero-shot-2 | 20002 | 0.334 | 0.975 | 0.908 | 0.933 | 0.832 | 0.919 |
| logistic | tfidf-grp 1000x5=5000 ng2-3 | zero-shot-2 | 20002 | 0.348 | 0.971 | 0.905 | 0.927 | 0.817 | 0.911 |
| logistic | tfidf-grp 100x5=500 ng1-2 | zero-shot-2 | 20002 | 0.460 | 0.945 | 0.884 | 0.882 | 0.709 | 0.832 |
| logistic | tfidf-grp 2000x5=10000 ng1-2 | zero-shot-2 | 20002 | 0.238 | 0.988 | 0.920 | 0.966 | 0.912 | 0.959 |
| logistic | tfidf-grp 200x5=1000 ng1-2 | zero-shot-2 | 20002 | 0.389 | 0.962 | 0.900 | 0.913 | 0.782 | 0.882 |
| logistic | tfidf-grp 500x5=2500 ng1-2 | zero-shot-2 | 20002 | 0.306 | 0.977 | 0.913 | 0.945 | 0.851 | 0.928 |
| xgboost | tfidf 1000 ng1-2 | zero-shot-2 | 20002 | 0.287 | 0.981 | 0.916 | 0.947 | 0.851 | 0.936 |
| xgboost | tfidf 10000 ng1-2 | zero-shot-2 | 20002 | 0.255 | 0.986 | 0.920 | 0.960 | 0.893 | 0.951 |
| xgboost | tfidf 2000 ng1-2 | zero-shot-2 | 20002 | 0.275 | 0.982 | 0.918 | 0.953 | 0.863 | 0.942 |
| xgboost | tfidf 500 ng1-2 | zero-shot-2 | 20002 | 0.310 | 0.977 | 0.914 | 0.937 | 0.820 | 0.924 |
| xgboost | tfidf 5000 ng1-1 | zero-shot-2 | 20002 | 0.250 | 0.987 | 0.920 | 0.960 | 0.896 | 0.951 |
| xgboost | tfidf 5000 ng1-2 | zero-shot-2 | 20002 | 0.257 | 0.985 | 0.920 | 0.957 | 0.883 | 0.949 |
| xgboost | tfidf 5000 ng1-3 | zero-shot-2 | 20002 | 0.263 | 0.985 | 0.919 | 0.956 | 0.881 | 0.948 |
| xgboost | tfidf 5000 ng2-2 | zero-shot-2 | 20002 | 0.397 | 0.964 | 0.902 | 0.897 | 0.775 | 0.898 |
| xgboost | tfidf 5000 ng2-3 | zero-shot-2 | 20002 | 0.407 | 0.962 | 0.899 | 0.893 | 0.762 | 0.893 |
| xgboost | tfidf-grp 1000x5=5000 ng1-1 | zero-shot-2 | 20002 | 0.253 | 0.986 | 0.919 | 0.959 | 0.891 | 0.948 |
| xgboost | tfidf-grp 1000x5=5000 ng1-2 | zero-shot-2 | 20002 | 0.253 | 0.986 | 0.920 | 0.959 | 0.891 | 0.949 |
| xgboost | tfidf-grp 1000x5=5000 ng1-3 | zero-shot-2 | 20002 | 0.254 | 0.986 | 0.920 | 0.959 | 0.887 | 0.948 |
| xgboost | tfidf-grp 1000x5=5000 ng2-2 | zero-shot-2 | 20002 | 0.371 | 0.969 | 0.905 | 0.910 | 0.814 | 0.904 |
| xgboost | tfidf-grp 1000x5=5000 ng2-3 | zero-shot-2 | 20002 | 0.379 | 0.968 | 0.903 | 0.908 | 0.805 | 0.898 |
| xgboost | tfidf-grp 100x5=500 ng1-2 | zero-shot-2 | 20002 | 0.335 | 0.974 | 0.909 | 0.926 | 0.783 | 0.902 |
| xgboost | tfidf-grp 2000x5=10000 ng1-2 | zero-shot-2 | 20002 | 0.243 | 0.987 | 0.921 | 0.962 | 0.901 | 0.954 |
| xgboost | tfidf-grp 200x5=1000 ng1-2 | zero-shot-2 | 20002 | 0.303 | 0.979 | 0.914 | 0.941 | 0.837 | 0.922 |
| xgboost | tfidf-grp 500x5=2500 ng1-2 | zero-shot-2 | 20002 | 0.272 | 0.983 | 0.918 | 0.953 | 0.873 | 0.940 |
