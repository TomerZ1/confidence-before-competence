# 01 — Data preparation (Tomer, run once, before anyone else)

**CPU notebook. Do NOT attach a GPU** — nothing here needs one, and it saves your GPU quota.
Internet must be ON. Runtime ~20–30 min (most of it is downloading the six `batch_indices` files).

## Run it

1. Kaggle → New Notebook → File → Import Notebook → `01_data_prep.ipynb`
2. Settings → Accelerator → **None**; Settings → Internet → **On**
3. Run All

## Then publish the probe set

The other three runners must score the **identical** probe set or nothing can be pooled.

1. In the notebook's Output pane, select `probes.json`, `exposure.npz`, `facts.parquet`
2. **Create Dataset** from the output, name it exactly **`lment-probes`**
3. Share it with Chen, Ofek and Victoria (or make it public)
4. Send them the **`PROBE_HASH`** the notebook printed

Their notebooks attach that dataset and assert the hash matches. If someone forgets to attach it,
the notebook regenerates the probes deterministically and prints its own hash — which must equal
yours. A mismatch means do not pool those results.

## What it produces

| File | Contents |
|---|---|
| `probes.json` | 3000 facts (300 × 10 exposure buckets), each with 10 length-matched candidates, the relation template, and its train/val/test split |
| `exposure.npz` | Cumulative exposure per fact at every 1000-step bin across all 6 epochs — how many subject-answer co-occurrences the model had actually seen by step *t* |
| `facts.parquet` | Fact metadata for those 3000 |

## Checklist before you hand off

- [ ] `14267 facts` loaded, 16 relations
- [ ] `3000 probes` built, 300 in every bucket
- [ ] gold vs distractor mean token length differ by < 0.5
- [ ] `chunk->step coverage` ≥ 99%
- [ ] the exposure assertion passed (`matches num_shared_chunks x 6 within 2%`)
- [ ] `PROBE_HASH` recorded and sent to the other three
- [ ] `lment-probes` dataset created and shared
