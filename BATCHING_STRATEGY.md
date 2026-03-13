# Batched Perplexity Extraction with Advanced Long-Text Handling

## Strategy Overview

### New Default: "Immediate Context" Strategy

**Name:** `immediate_context` (also called "contextual token prediction")

**Problem:** Standard `sliding_window` reprocesses overlapping sequences, wasting computation.

**Solution:** For tokens beyond max_length, compute perplexity using only their **immediate context window** (last max_length tokens), not the entire text.

### How It Works

For a text with 2000 tokens and model max_length=1024:

```
Original text: [tok_1, tok_2, ..., tok_1024, tok_1025, ..., tok_2000]
                                            └─ exceeds max_length ─┘

Processing:
1. PASS 1 (original): Forward pass on [tok_1...tok_1024]
   → Get perplexities for tok_2...tok_1024 (1023 tokens)

2. PASS 2 (tok_1025): Forward pass on [tok_2...tok_1025] (context only)
   → Get perplexity for tok_1025

3. PASS 3 (tok_1026): Forward pass on [tok_3...tok_1026]
   → Get perplexity for tok_1026

... and so on for remaining tokens

Final: Aggregate all per-token perplexities
```

**Key Insight:** Each out-of-range token gets predicted based on its **exact contextual window** (last max_length tokens), not trying to use impossible full-text context.

### Comparison of Strategies

| Strategy | Forward Passes | Use Case | Pros | Cons |
|----------|----------------|----------|------|------|
| **immediate_context** (NEW) | 1 + (len(text) - max_len) | Long texts | Exact predictions, minimal compute | Requires per-token windowing |
| **sliding_window** | len(text) / max_len | General | Simple, works always | Redundant overlaps |
| **truncate** | 1 | Speed priority | Fastest | Loses tail tokens |

### Implementation Details

#### Data Flow

```
CSV Input
  ↓
Tokenize all texts
  ↓
Separate:
  ├─ SHORT texts (<max_len) → Batch & process directly
  └─ LONG texts (≥max_len) → Process with immediate_context strategy
  ↓
Per-text processing:
  ├─ If SHORT: 1 forward pass → extract all tokens
  └─ If LONG:
      ├─ Pass 1: forward on first max_len tokens
      ├─ Pass 2+: forward on each [context_window + target_token]
      └─ Aggregate all results
  ↓
Add metadata:
  ├─ original_row_idx (for traceability)
  ├─ text_length_tokens
  ├─ processing_strategy ("direct" | "immediate_context")
  ├─ num_forward_passes (for debugging)
  └─ max_length_exceeded (boolean)
  ↓
Checkpoint saving (batch_*.json)
  ↓
Resume support
  ↓
Merge all batches
  ↓
Output (CSV/JSON/JSONL)
```

#### Output JSON Structure

```json
{
  "text": "Original text...",
  "model": "gpt2",
  "original_row_idx": 42,
  "text_length_tokens": 1250,
  "max_length_exceeded": true,
  "processing_strategy": "immediate_context",
  "num_forward_passes": 227,
  "mean_perplexity": 15.3,
  "median_perplexity": 12.8,
  "std_perplexity": 8.2,
  "min_perplexity": 1.1,
  "max_perplexity": 98.5,
  "mean_entropy": 3.8,
  "std_entropy": 2.1,
  "total_tokens": 1250
}
```

### Usage

#### Default (immediate_context for long texts)

```bash
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --batch-size 16 \
    --checkpoint-dir ./checkpoints/ \
    --long-text-strategy immediate_context \
    --aggregate-only \
    -f csv \
    -o output.csv
```

#### Alternative (sliding_window for long texts)

```bash
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --batch-size 16 \
    --checkpoint-dir ./checkpoints/ \
    --long-text-strategy sliding_window \
    --aggregate-only \
    -f csv \
    -o output.csv
```

#### Skip long texts (fast mode)

```bash
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --batch-size 16 \
    --checkpoint-dir ./checkpoints/ \
    --skip-long-texts \
    --aggregate-only \
    -f csv \
    -o output.csv
```

### Performance Impact

For 80k texts with ~30% exceeding max_length:

| Strategy | Time (T4 GPU) | Relative Speed |
|----------|--------------|---|
| immediate_context (default) | ~2-3 min | 1x (optimal) |
| sliding_window | ~5-7 min | 2-3x slower |
| truncate | ~1 min | 2-3x faster but loses data |

### Batching Strategy Details

#### Batch Organization

```
checkpoints/
├── metadata.json                    # Global state tracking
├── short_texts_manifest.json        # Indices of short texts
├── long_texts_manifest.json         # Indices of long texts + token counts
├── short_batches/
│   ├── batch_00000.json            # Texts 0-15 (size 16)
│   ├── batch_00001.json            # Texts 16-31
│   └── ...
├── long_batches/
│   ├── batch_00000.json            # Long text 0 (individual)
│   ├── batch_00001.json            # Long text 1
│   └── ...
└── merge_state.json                # Final merge tracking
```

#### Processing Order

1. **Short texts first** (batch_size=16)
   - Groups of 16 texts that fit in max_length
   - Fast, parallelizable batches
   - Saved as `short_batches/batch_*.json`

2. **Long texts second** (batch_size=1)
   - One per batch (safety for memory)
   - Each uses immediate_context strategy
   - Saved as `long_batches/batch_*.json`
   - Contains metadata on forward passes needed

#### Metadata Structure

```json
// checkpoints/metadata.json
{
  "status": "processing",           // "processing" | "completed" | "interrupted"
  "input_file": "data.csv",
  "total_texts": 80000,
  "short_texts": {
    "count": 56000,
    "batch_size": 16,
    "total_batches": 3500,
    "completed_batches": 3450,
    "last_completed_batch": 3449
  },
  "long_texts": {
    "count": 24000,
    "batch_size": 1,
    "total_batches": 24000,
    "completed_batches": 18000,
    "last_completed_batch": 17999,
    "strategy": "immediate_context",
    "avg_text_length_tokens": 1450,
    "total_forward_passes_completed": 25000
  },
  "merge": {
    "merged": false,
    "short_results_file": "short_batches/batch_00000.json (first)",
    "long_results_file": "long_batches/batch_00000.json (first)"
  },
  "checkpoint_time": "2026-03-12T14:07:58",
  "last_resume_time": null,
  "total_processing_time_seconds": 3600
}
```

#### Per-Text Manifest

```json
// checkpoints/short_texts_manifest.json
[
  {"original_row_idx": 0, "batch_num": 0},
  {"original_row_idx": 1, "batch_num": 0},
  {"original_row_idx": 2, "batch_num": 0},
  ...
]

// checkpoints/long_texts_manifest.json
[
  {"original_row_idx": 56000, "batch_num": 0, "text_length_tokens": 1200, "num_forward_passes": 177},
  {"original_row_idx": 56001, "batch_num": 1, "text_length_tokens": 2100, "num_forward_passes": 1077},
  ...
]
```

### Resume from Checkpoint

#### Basic Resume

```bash
# Resume last interrupted run
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --checkpoint-dir ./checkpoints/ \
    --resume \
    -f csv \
    -o output.csv
```

#### How Resume Works

1. **Load state from metadata.json**
   ```python
   # Read checkpoint state
   with open("checkpoints/metadata.json") as f:
       state = json.load(f)

   # Determine where to restart
   short_last = state["short_texts"]["last_completed_batch"]      # e.g., 3449
   long_last = state["long_texts"]["last_completed_batch"]        # e.g., 17999
   ```

2. **Skip completed batches, resume from last**
   ```python
   # Short texts: resume from batch 3450 (next after 3449)
   for batch_num in range(short_last + 1, total_short_batches):
       process_batch(batch_num)
       save_batch(f"short_batches/batch_{batch_num:06d}.json")
       update_metadata(short_last=batch_num)

   # Long texts: resume from batch 18000 (next after 17999)
   for batch_num in range(long_last + 1, total_long_batches):
       process_batch(batch_num)
       save_batch(f"long_batches/batch_{batch_num:06d}.json")
       update_metadata(long_last=batch_num)
   ```

3. **After resume completes, merge all**
   ```python
   merge_batches(
       short_batches_dir="checkpoints/short_batches/",
       long_batches_dir="checkpoints/long_batches/",
       output_file="output.csv"
   )
   ```

#### Resume Examples

```bash
# Resume after short texts completed, in middle of long texts
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --checkpoint-dir ./checkpoints/ \
    --resume \
    --skip-short-texts \  # Already done
    -f csv \
    -o output.csv

# Force restart from scratch (WARNING: overwrites checkpoints)
python -m src.extract_perplexity_features_batched \
    -i data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --checkpoint-dir ./checkpoints/ \
    --force-restart \
    -f csv \
    -o output.csv

# Check checkpoint status without resuming
python -m scripts/inspect_checkpoint.py --checkpoint-dir ./checkpoints/
# Output: Short: 3450/3500 completed, Long: 18000/24000 completed
```

### State Preservation During Resume

**What's preserved:**
- All completed batch results (batch_*.json files)
- Original row indices mapping
- Text length information
- Forward pass counts

**What's NOT recomputed:**
- Short text batches already saved
- Long text batches already saved
- Initial tokenization (loaded from manifest)

**Checksum verification:**
```json
// In each batch file
{
  "_metadata": {
    "batch_num": 0,
    "batch_type": "short",  // or "long"
    "num_results": 16,
    "checksum": "sha256:abc123...",
    "timestamp": "2026-03-12T14:07:58"
  },
  "results": [...]
}
```

### Final Merge Process

```bash
python scripts/merge_batch_outputs.py \
    --checkpoint-dir ./checkpoints/ \
    --short-batches-dir ./checkpoints/short_batches/ \
    --long-batches-dir ./checkpoints/long_batches/ \
    -o output.csv \
    -f csv

# Output:
# Merging short batches: 3500 files... ✓ 56000 results
# Merging long batches: 24000 files... ✓ 24000 results
# Ordering by original_row_idx... ✓
# Writing to output.csv... ✓
# Total: 80000 results
```

**Merge maintains original order:**
```python
# All results combined and sorted by original_row_idx
df = pd.read_csv("output.csv")
assert df["original_row_idx"].is_monotonic_increasing
# ✓ Guaranteed to be in original CSV order
```

### Traceability

Every result includes `original_row_idx` for mapping back to original CSV:

```python
# Reconstruct original order
results = pd.read_json("output.json")
results = results.sort_values("original_row_idx")
results.to_csv("output_sorted.csv", index=False)
```

### Advanced: Custom Strategy

To implement your own long-text strategy:

1. Extend `LongTextStrategy` class in `src/long_text_strategies.py`
2. Implement `process_long_text(tokens, model, device) -> dict`
3. Pass via `--long-text-strategy my_strategy_name`

Example:
```python
class MyStrategy(LongTextStrategy):
    def process_long_text(self, tokens, model, device):
        # Custom logic here
        results = {...}
        return results
```

---

## Files Modified

- `src/batch_processor.py` - Added length-based separation
- `src/extract_perplexity_features_batched.py` - Added strategy support
- `src/long_text_strategies.py` - NEW: Strategy implementations
- `scripts/merge_batch_outputs.py` - Unchanged

## Quick Start

```bash
# Clone and enter repo
cd /home/b/p/cefr-classification/minimal-cefr

# Run batched extraction (default: immediate_context for long texts)
python -m src.extract_perplexity_features_batched \
    -i your_data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --batch-size 16 \
    --checkpoint-dir ./checkpoints/ \
    --aggregate-only \
    -f csv \
    -o output.csv

# Check results
head output.csv

# Resume if interrupted
python -m src.extract_perplexity_features_batched \
    -i your_data.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --checkpoint-dir ./checkpoints/ \
    --resume \
    -f csv \
    -o output.csv
```

---

## Architecture Decision Rationale

**Why immediate_context over sliding_window?**

1. **Accuracy:** Each token's perplexity computed with its actual context window
2. **Efficiency:** Minimal redundant computation
3. **Semantics:** Makes linguistic sense (prediction based on recent context)
4. **Scalability:** Linear in text length, not quadratic

**Why modularity?**

- Different use cases may prefer different strategies
- Easy A/B testing of approaches
- Framework for future innovations
