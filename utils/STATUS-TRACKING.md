# Pipeline Status Tracking

## Overview

The pipeline test script now automatically tracks all steps in a JSONL (JSON Lines) status file located at `{experiment_folder}/__status.jsonl`.

## Status File Format

Each line in `__status.jsonl` is a JSON object with the following structure:

```json
{"timestamp":"2025-10-19T10:30:45+0000","step":"step_name","status":"success","message":"Step completed successfully"}
```

### Fields

- **timestamp**: ISO 8601 timestamp when the event occurred
- **step**: Name of the pipeline step (function name)
- **status**: One of: `success`, `error`, `warning`, `started`, `skipped`
- **message**: Human-readable description of what happened

## Status Values

| Status | Meaning | Exit Behavior |
|--------|---------|---------------|
| `started` | Step has begun execution | Continues |
| `success` | Step completed successfully | Continues |
| `error` | Step failed | Pipeline stops |
| `warning` | Non-fatal issue occurred | Continues |
| `skipped` | Step was skipped (via `--skip-*` flag) | Continues |

## Tracked Steps

The following pipeline steps are tracked:

1. **setup_experiment_directory** - Create experiment directory structure
2. **generate_synthetic_data** - Generate fake CEFR training/test data
3. **create_pipeline_config** - Create config.yaml file
4. **run_tfidf_training** - Train TF-IDF vectorizer
5. **run_feature_extraction** - Extract features from text
6. **run_classifier_training** - Train classifier model
7. **run_prediction** - Run predictions on test data
8. **pipeline_complete** - Overall pipeline completion

## Example Status File

```jsonl
{"timestamp":"2025-10-19T10:30:00+0000","step":"setup_experiment_directory","status":"started","message":"Setting up experiment directory"}
{"timestamp":"2025-10-19T10:30:01+0000","step":"setup_experiment_directory","status":"success","message":"Experiment directory created successfully"}
{"timestamp":"2025-10-19T10:30:01+0000","step":"generate_synthetic_data","status":"started","message":"Starting synthetic data generation"}
{"timestamp":"2025-10-19T10:30:05+0000","step":"generate_synthetic_data","status":"success","message":"Generated 50 training and 20 test samples"}
{"timestamp":"2025-10-19T10:30:05+0000","step":"create_pipeline_config","status":"started","message":"Creating pipeline configuration file"}
{"timestamp":"2025-10-19T10:30:06+0000","step":"create_pipeline_config","status":"success","message":"Configuration file created successfully"}
{"timestamp":"2025-10-19T10:30:06+0000","step":"run_tfidf_training","status":"started","message":"Starting TF-IDF model training"}
{"timestamp":"2025-10-19T10:30:15+0000","step":"run_tfidf_training","status":"success","message":"TF-IDF model trained successfully"}
{"timestamp":"2025-10-19T10:30:15+0000","step":"run_feature_extraction","status":"started","message":"Starting feature extraction"}
{"timestamp":"2025-10-19T10:30:25+0000","step":"run_feature_extraction","status":"success","message":"Features extracted for training and test data"}
{"timestamp":"2025-10-19T10:30:25+0000","step":"run_classifier_training","status":"started","message":"Starting logistic classifier training"}
{"timestamp":"2025-10-19T10:30:35+0000","step":"run_classifier_training","status":"success","message":"logistic classifier trained successfully"}
{"timestamp":"2025-10-19T10:30:35+0000","step":"run_prediction","status":"started","message":"Starting prediction on test data"}
{"timestamp":"2025-10-19T10:30:40+0000","step":"run_prediction","status":"success","message":"Predictions completed successfully"}
{"timestamp":"2025-10-19T10:30:40+0000","step":"pipeline_complete","status":"success","message":"All pipeline steps completed successfully"}
```

## Status File Location

Default location: `{experiment_folder}/__status.jsonl`

Example:
```
test-experiments/test-pipeline-20251019_103000/__status.jsonl
```

The double underscore prefix (`__`) indicates this is a metadata/log file.

## Checking Pipeline Status

### Method 1: Automatic Summary (during execution)

The pipeline script automatically displays a status summary at the end:

```
Status Summary:
  Total events: 14
  Successes: 7
  Errors: 0
  Skipped: 0
```

### Method 2: Using the Status Checker Script

```bash
# Check most recent pipeline run
./utils/check-pipeline-status.sh

# Check specific status file
./utils/check-pipeline-status.sh test-experiments/test-pipeline-20251019_103000/__status.jsonl
```

The checker script provides:
- Summary statistics
- Overall status (SUCCESS/FAILED/INCOMPLETE)
- Detailed timeline with color-coded events
- Critical steps verification
- Exit code (0=success, 1=failed, 2=incomplete)

### Method 3: Manual Inspection

```bash
# View raw status file
cat test-experiments/test-pipeline-20251019_103000/__status.jsonl

# Pretty print with jq (if available)
cat test-experiments/test-pipeline-20251019_103000/__status.jsonl | jq .

# Check for errors
grep '"status":"error"' test-experiments/test-pipeline-20251019_103000/__status.jsonl

# Count successful steps
grep -c '"status":"success"' test-experiments/test-pipeline-20251019_103000/__status.jsonl

# Verify all steps succeeded
grep '"status":"success"' test-experiments/test-pipeline-20251019_103000/__status.jsonl | wc -l
```

### Method 4: Programmatic Checking

```bash
#!/bin/bash
# Check if all critical steps succeeded

STATUS_FILE="test-experiments/latest/__status.jsonl"

critical_steps=(
    "run_tfidf_training"
    "run_feature_extraction"
    "run_classifier_training"
    "run_prediction"
)

all_success=true

for step in "${critical_steps[@]}"; do
    if ! grep -q "\"step\":\"$step\".*\"status\":\"success\"" "$STATUS_FILE"; then
        echo "Step $step did not succeed"
        all_success=false
    fi
done

if [[ "$all_success" == "true" ]]; then
    echo "All steps succeeded!"
    exit 0
else
    echo "Some steps failed!"
    exit 1
fi
```

## Verifying Complete Success

To verify that **all** pipeline steps completed successfully:

```bash
# All success steps should equal number of critical steps
# (setup, data_gen, config, tfidf, features, classifier, prediction, complete)

SUCCESS_COUNT=$(grep -c '"status":"success"' __status.jsonl)

# Should be at least 8 for complete pipeline
if [[ $SUCCESS_COUNT -ge 8 ]]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline incomplete or failed"
fi
```

Or simply check for the final success marker:

```bash
# Check if pipeline_complete was logged as success
if grep -q '"step":"pipeline_complete".*"status":"success"' __status.jsonl; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline did not complete"
fi
```

## Error Investigation

When errors occur, the status file helps identify exactly where the pipeline failed:

```bash
# Find all errors
grep '"status":"error"' __status.jsonl

# Get error messages
grep '"status":"error"' __status.jsonl | jq -r '.message'

# Find which step failed first
grep '"status":"error"' __status.jsonl | head -1 | jq -r '.step'
```

## Integration with CI/CD

Use the status checker in continuous integration:

```bash
#!/bin/bash
# CI/CD pipeline test

# Run pipeline test
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name ci-test \
    --cleanup \
    --quiet

# Check status (exit code indicates success/failure)
./utils/check-pipeline-status.sh

# Exit code propagates to CI/CD system
```

## Debugging with Status File

The status file is especially useful for debugging:

1. **Identify failure point**: See exactly which step failed
2. **Check timing**: See how long each step took
3. **Verify skips**: Confirm which steps were intentionally skipped
4. **Compare runs**: Diff status files from different runs

Example comparison:

```bash
# Compare two runs
diff -u \
    test-experiments/run1/__status.jsonl \
    test-experiments/run2/__status.jsonl
```

## Status File Retention

The status file is kept with the experiment data:

- **Not cleaned up** even with `--cleanup` flag
- **Preserved** for post-mortem analysis
- **Small size** (typically < 1KB)
- **Human-readable** JSONL format

## Parsing Status File

### Using jq (recommended)

```bash
# Get all successful steps
jq -r 'select(.status=="success") | .step' __status.jsonl

# Get error messages
jq -r 'select(.status=="error") | "\(.step): \(.message)"' __status.jsonl

# Get timeline
jq -r '"\(.timestamp) [\(.status)] \(.step)"' __status.jsonl
```

### Using Python

```python
import json

with open('__status.jsonl') as f:
    events = [json.loads(line) for line in f]

# Find errors
errors = [e for e in events if e['status'] == 'error']

# Check if all succeeded
all_success = all(
    any(e['step'] == step and e['status'] == 'success' for e in events)
    for step in ['run_tfidf_training', 'run_feature_extraction',
                 'run_classifier_training', 'run_prediction']
)
```

### Using grep/awk

```bash
# Extract step names that succeeded
grep '"status":"success"' __status.jsonl | \
    awk -F'"' '{for(i=1;i<=NF;i++){if($i=="step"){print $(i+2)}}}'

# Count events by status
awk -F'"' '{for(i=1;i<=NF;i++){if($i=="status"){status[$(i+2)]++}}}
            END{for(s in status){print s":",status[s]}}' __status.jsonl
```

## Best Practices

1. **Always check status file** after pipeline runs
2. **Keep status files** for successful runs (small, useful for reference)
3. **Review errors immediately** using status file
4. **Automate checks** in CI/CD using exit codes
5. **Compare status files** when debugging behavioral changes
6. **Archive status files** with experiment results

## Exit Codes from check-pipeline-status.sh

| Exit Code | Meaning |
|-----------|---------|
| 0 | All steps succeeded |
| 1 | One or more steps failed |
| 2 | Pipeline incomplete (some steps not run) |

Use in scripts:

```bash
if ./utils/check-pipeline-status.sh "$STATUS_FILE"; then
    echo "Pipeline succeeded!"
else
    echo "Pipeline failed!"
fi
```

## Files

- **Main script**: `utils/minimal-fake-example-and-pipeline.sh`
- **Status checker**: `utils/check-pipeline-status.sh`
- **Status file**: `{experiment_dir}/__status.jsonl`
- **This guide**: `utils/STATUS-TRACKING.md`
