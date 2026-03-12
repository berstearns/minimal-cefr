# Pre-extracted Perplexity Feature Files

All files live under `/home/b/p/cefr-classification/gdrive-data/fe/`.

## File Format

Files are gzip-compressed CSVs (`.csv.features.gzip`). Two variants exist:

| Suffix | Headers | Columns | Notes |
|---|---|---|---|
| `avg+ppp` | numeric values as headers | 513 | avg_loss + 512 per-position perplexity |
| `avg+ppp+ppos` | proper semantic names | 555 | above + ~41 per-POS-tag avg perplexity |
| `avg+ppp+ppos-flat` | numeric values as headers (bug) | 554 | same content, lost column names |

**Use the non-flat `avg+ppp+ppos` variant** when available -- it has proper column
names like `measures.perplexity.wholetext.gpt2.position.token_1`.

## What the Features Represent

Each text is fed through a causal language model (GPT-2 or an Artificial Learner
fine-tuned from GPT-2). The model predicts each token given the preceding context.
From these predictions, three kinds of features are extracted:

### 1. Per-Position Perplexity (PPP) -- 512 columns

Column names: `measures.perplexity.wholetext.<model>.position.token_{1..512}`

For each token position in the text, the **token-level cross-entropy loss** is
recorded. This is the negative log-probability the model assigned to the actual
next token at that position:

    loss_i = -log P(token_i | token_1, ..., token_{i-1})

Values are in **nats** (natural log). Typical range: 0.01 (highly predictable
tokens like "the", "is") to 20 (surprising or rare tokens).

Texts shorter than 512 tokens are **zero-padded** on the right. Texts longer
than 512 are truncated (GPT-2's max context window).

**Linguistic intuition**: A1 learners produce simpler, more predictable token
sequences (lower per-position loss under a native model). C1 learners use rarer
vocabulary and more complex syntax (higher loss at certain positions). The
position dimension captures where in a text the surprisal changes -- e.g., early
tokens tend to have higher loss (less context), while later tokens stabilize.

### 2. Average Loss -- 1 column

Column name: `measures.perplexity.wholetext.<model>.avg_loss`

The **mean cross-entropy loss** across all non-padded tokens in the text:

    avg_loss = (1/N) * sum(loss_i for i in 1..N)

where N is the actual token count (before padding). This is the single most
compact summary of how "surprising" the text is to the model.

**Text-level perplexity** = exp(avg_loss). For example, avg_loss=4.04 corresponds
to perplexity=56.8, meaning the model is as uncertain as if choosing uniformly
among ~57 tokens at each step.

### 3. Per-POS Perplexity (PPOS) -- 41 columns

Column names: `measures.perplexity.wholetext.<model>.pos.<TAG>`

Each token is POS-tagged (Penn Treebank tagset). The per-position losses are then
**summed by POS tag**:

    pos_NN = sum(loss_i for all tokens i where POS(token_i) == "NN")

This means the POS values are **not averages** -- they are sums. A tag that
appears more frequently in the text accumulates a higher total. Tags absent from
the text get value 0.

Verified property: `sum(all position values) == sum(all POS tag values)` -- the
POS features are a re-partitioning of the same per-token losses, grouped by
grammatical category instead of position.

**The 41 POS tags:**

| Tag | Category | Example |
|---|---|---|
| NN, NNS, NNP, NNPS | Nouns | cat, cats, London, Alps |
| VB, VBD, VBG, VBN, VBP, VBZ | Verbs | run, ran, running, run, run, runs |
| JJ, JJR, JJS | Adjectives | big, bigger, biggest |
| RB, RBR, RBS | Adverbs | quickly, faster, fastest |
| DT, PDT | Determiners | the, both |
| IN | Preposition/conjunction | in, of, that |
| PRP, PRP$ | Pronouns | he, his |
| CC | Coordinating conjunction | and, but |
| CD | Cardinal number | 3, seven |
| MD | Modal | can, should |
| TO | "to" | to |
| EX | Existential "there" | there |
| WDT, WP, WP$, WRB | Wh-words | which, who, whose, where |
| RP | Particle | up (in "give up") |
| FW | Foreign word | ad hoc |
| LS | List marker | 1., a) |
| SYM | Symbol | $, % |
| UH | Interjection | oh, wow |
| POS | Possessive ending | 's |
| `.` `(` `)` `''` `$` | Punctuation | . ( ) '' $ |

**Linguistic intuition**: Low-level learners tend to accumulate high total
perplexity on verb forms (irregular verbs, tense errors) and nouns (limited
vocabulary). The POS features let the classifier distinguish "surprising nouns"
from "surprising verbs" -- important for proficiency assessment because error
patterns differ systematically across CEFR levels.

### How Multi-Model Features Work

When using multiple models (e.g., native GPT-2 + 5 Artificial Learners), the
same text is run through each model independently, producing 554 features per
model. These are **column-concatenated** into a single feature vector.

For 7 models: 7 x 554 = 3,878 features per text.

The classifier learns which model's surprisal at which position/POS best
discriminates CEFR levels. For example:
- An A1 text might have LOW loss under the A1-AL (it's seen similar texts) but
  HIGH loss under the native GPT-2 (poor English is surprising to a native model)
- A C1 text might have LOW loss under native GPT-2 (fluent English) but HIGH
  loss under the A1-AL (advanced vocabulary is rare in A1 training data)

## Column Structure Summary (554 features per model)

| Feature group | Count | Description |
|---|---|---|
| `...position.token_{1..512}` | 512 | Per-token cross-entropy loss at each position (nats, zero-padded) |
| `...avg_loss` | 1 | Mean cross-entropy loss across non-padded tokens |
| `...pos.{POS_TAG}` | 41 | Sum of per-token losses grouped by POS tag |

## Models Available

| Model ID | Description | Training data |
|---|---|---|
| `gpt2` | Pre-trained GPT-2 (native English) | OpenAI pre-training |
| `AL-all-gpt2` | General Artificial Learner | All EFCAMDAT remainder (~623k) |
| `AL-a1-gpt2` | A1-level AL | EFCAMDAT remainder A1 subset |
| `AL-a2-gpt2` | A2-level AL | EFCAMDAT remainder A2 subset |
| `AL-b1-gpt2` | B1-level AL | EFCAMDAT remainder B1 subset |
| `AL-b2-gpt2` | B2-level AL | EFCAMDAT remainder B2 subset |
| `AL-c1-gpt2` | C1-level AL | EFCAMDAT remainder C1 subset |

## Dataset Mapping

The feature files use legacy dataset names. Mapping to the pipeline's names:

| Feature file prefix | Pipeline dataset name | Role | Rows |
|---|---|---|---|
| `andrew100ktrain_df` | `norm-EFCAMDAT-train` | Training set | ~80,000 |
| `andrew100ktest_df` | `norm-EFCAMDAT-test` | In-domain test | ~20,000 |
| `celva` | `norm-CELVA-SP` | Cross-corpus test | 1,742 |
| `KUPA-KEYS` | `norm-KUPA-KEYS` | Cross-corpus test | 1,006 |

---

## Experiment 1: Native GPT-2 Only

Single model, simplest baseline.

**Training features (EFCAMDAT-train):**
```
gdrive-data/fe/andrew100ktrain_df-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
```

**Test features:**
```
gdrive-data/fe/andrew100ktest_df-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/celva-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
```

Columns: 554 (1 model x 554 features)

---

## Experiment 2: Native GPT-2 + General AL

Two models combined. Tests whether the general learner trajectory adds signal.

**Training features (EFCAMDAT-train):**
```
gdrive-data/fe/andrew100ktrain_df-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-13-31.csv.features.gzip
```

**Test features:**
```
gdrive-data/fe/andrew100ktest_df-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip

gdrive-data/fe/celva-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip

gdrive-data/fe/KUPA-KEYS-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
```

Columns: ~1108 (2 models x 554 features, column-concatenated)

---

## Experiment 3: All 7 Models (Full Paper Configuration)

Native GPT-2 + General AL + 5 level-specific ALs. This is the full paper setup.

**Training features (EFCAMDAT-train):**
```
gdrive-data/fe/andrew100ktrain_df-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-13-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
gdrive-data/fe/andrew100ktrain_df-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip
```

**Test features (EFCAMDAT-test):**
```
gdrive-data/fe/andrew100ktest_df-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-a1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-a2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-b1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-b2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-c1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
```

**Test features (CELVA-SP):**
```
gdrive-data/fe/celva-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/celva-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip
```

**Test features (KUPA-KEYS):**
```
gdrive-data/fe/KUPA-KEYS-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/KUPA-KEYS-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip
```

Columns: ~3878 (7 models x 554 features, column-concatenated)

---

## Experiment 4: Pre-concatenated All-Models (Flat)

These files already have all 7 models concatenated in a single file:

```
gdrive-data/fe/KUPA-KEYS-AL-a1-gpt2+AL-a2-gpt2+AL-b1-gpt2+AL-b2-gpt2+AL-c1-gpt2+gpt2+AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-23-06-04.csv.features.gzip
gdrive-data/fe/celva-AL-a1-gpt2+AL-a2-gpt2+AL-b1-gpt2+AL-b2-gpt2+AL-c1-gpt2+gpt2+AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-23-08-31.csv.features.gzip
gdrive-data/fe/andrew100ktest_df-AL-a1-gpt2+AL-a2-gpt2+AL-all-gpt2+AL-b1-gpt2+AL-b2-gpt2+AL-c1-gpt2+gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip
```

Note: no pre-concatenated file exists for `andrew100ktrain`. The training set
must be assembled by column-concatenating the 7 individual model files.

Columns: ~3878 (flat, no semantic column names)

---

## Recommendation

**Most promising for the paper**: Experiment 3 (all 7 models, non-flat) gives
the best results and has proper column names for interpretability. If the
non-flat variant is unavailable for a dataset, fall back to the flat version
and assign positional column names.

**Quickest win**: Experiment 1 (native GPT-2 only) runs in seconds and
establishes the baseline.
