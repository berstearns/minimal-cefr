# Quick Start: Experiment Structure Validation

## Check Your Experiment Structure

```bash
# Validate your experiment directory
python -m src.experiment_structure validate data/experiments/zero-shot-2
```

**Expected output if valid:**
```
✓ Experiment structure validated: data/experiments/zero-shot-2
```

**Expected output if invalid:**
```
Experiment structure validation for: data/experiments/my-experiment
✗ Errors (2):
  - Required directory missing: ml-training-data/
  - Required directory missing: features/
```

## Create a New Experiment Structure

```bash
# Create all required directories
python -m src.experiment_structure create data/experiments/my-new-experiment
```

This creates:
- `ml-training-data/`
- `ml-test-data/`
- `feature-models/`
- `feature-models/classifiers/`
- `features/`
- `results/`

## View Full Structure Documentation

```bash
# Show detailed structure documentation
python -m src.experiment_structure show
```

## Verify All Scripts Are Properly Documented

```bash
# Check that all scripts have structure documentation
python utils/validate_structure_docs.py
```

**Current status:**
- ✅ `predict.py` - Fully documented with validation
- ⚠️ Other scripts - Need updates (see STRUCTURE_VALIDATION.md)

## Running Predictions with Structure Validation

The `predict.py` script now validates structure automatically:

```bash
# Structure will be validated before predictions run
python -m src.predict \
  -e data/experiments/zero-shot-2 \
  --batch-features-dir data/experiments/zero-shot-2/features \
  --batch-models-dir data/experiments/zero-shot-2/feature-models/classifiers \
  --cefr-column cefr_level
```

If structure is invalid, you'll see:
```
✗ Experiment structure validation failed!
  Required directory missing: ml-test-data/

See docstring or src/experiment_structure.py for expected structure.
```

## Troubleshooting

### "Required directory missing" error

1. Check the structure with: `python -m src.experiment_structure validate <dir>`
2. Create missing directories manually or use: `python -m src.experiment_structure create <dir>`

### Need to customize required directories

In your script's `main()` function:

```python
# Only validate directories this script needs
required_dirs = ['features', 'feature-models']  # Customize as needed
is_valid, errors = validate_experiment_structure(
    experiment_dir,
    required_dirs=required_dirs,
    verbose=True
)
```

## Reference Files

- **Structure definition**: `src/experiment_structure.py`
- **Example implementation**: `src/predict.py` (see docstring and main())
- **Full documentation**: `STRUCTURE_VALIDATION.md`
