# Data Setup

## Required Files

Four CSV files are needed, all with the same schema:

| File | Rows | Role |
|------|------|------|
| `norm-EFCAMDAT-train.csv` | 80,000 | Training set |
| `norm-EFCAMDAT-test.csv` | 20,000 | Test set (in-domain) |
| `norm-CELVA-SP.csv` | 1,742 | Test set (cross-corpus) |
| `norm-KUPA-KEYS.csv` | 1,006 | Test set (cross-corpus) |

## CSV Schema

```
writing_id,l1,cefr_level,text
12345,Arabic,B1,"The student wrote about their experience..."
```

Required columns:
- `text` - the learner text (used for perplexity extraction)
- `cefr_level` - CEFR level label: A1, A2, B1, B2, C1, or C2

Other columns (`writing_id`, `l1`) are preserved but not used by the pipeline.

## Option A: Google Drive

1. Ensure folder exists in Google Drive: `MyDrive/phd-experimental-data/data/splits/`
2. Sync your rclone data there using:
   ```bash
   rclone sync i:/phd-experimental-data/data/splits/ gdrive:phd-experimental-data/data/splits/
   ```
3. In the notebook, the default path is `/content/drive/MyDrive/phd-experimental-data/data/splits`
4. Change the `DATA_PATH` variable in the notebook if you use a different location

## Option B: Download via URL

1. Package the 4 CSVs into a .zip file
2. Host the .zip at a direct-download URL
3. In the notebook, set `DATA_URL` to your URL
4. After extraction, adjust `DATA_PATH` to match the zip's internal structure

Expected zip structure:
```
cefr-data.zip
└── splits/
    ├── norm-EFCAMDAT-train.csv
    ├── norm-EFCAMDAT-test.csv
    ├── norm-CELVA-SP.csv
    └── norm-KUPA-KEYS.csv
```

If your zip has a different structure (e.g., files at the root), update
`DATA_PATH` accordingly:
```python
DATA_PATH = Path('/content/data')  # if CSVs are at zip root
```
