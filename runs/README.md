# 02 — Checkpoint sweeps (all four runners, in parallel)

Do not start until Tomer has published the `lment-probes` dataset from `01_data_prep`.

Every folder contains the **same notebook** with a different `PERSON` set at the top, so the
scoring code is byte-identical across all four of you. Only the assignment differs.

| Person | Models and checkpoints | Runs | Est. |
|---|---|---|---|
| [Tomer](tomer/) | `LMEnt-1B-6E` step0–10000, dense | 11 | ~2.0 h |
| [Chen](chen/) | `LMEnt-1B-6E` 20k–170k, + `LMEnt-170M-6E` matched | 19 | ~1.7 h |
| [Ofek](ofek/) | `LMEnt-1B-6E` 220k–658k, + `LMEnt-600M-6E` matched | 19 | ~2.4 h |
| [Victoria](victoria/) | `LMEnt-1B-1E` all, + template-robustness on `1B-6E` | 18 | ~2.7 h |

## Why the split is shaped this way

The pilot showed **88% of the confidence rise happens by step 1000**, and `LMEnt-1B-6E` is the
only released model with checkpoints below step 10000. So Tomer takes that dense early window —
it is where the phenomenon actually forms. Chen and Ofek split the long plateau and each also
run one smaller model, giving a 170M/600M/1B scale axis at matched steps. Victoria runs the
1-epoch model, which sees the same data once rather than six times and so separates *repetition*
from *volume*, plus two paraphrased template sets to show the result is not a wording artifact.

## Rules that matter

- **Accelerator: `GPU T4 x2`.** P100 is sm_60 and current PyTorch has no kernels for it — it
  crashes. The notebook checks and stops with instructions if you pick wrong.
- **Internet: On.**
- **Attach `lment-probes`** (Add Data → search it). Otherwise the notebook regenerates the probe
  set and you must verify the printed `PROBE_HASH` matches Tomer's.
- **fp32 is deliberate.** LMEnt weights are fp32 and the 1B model's activations reach ~3.1e6,
  47× past fp16's ceiling — fp16 produces all-NaN logits. Do not "optimise" this to fp16.
- **Resumable.** Results save after every checkpoint and completed ones are skipped. If Kaggle
  kills the session, just Run All again.

## Send back to Tomer

Every `res__*.parquet` plus your `manifest__<name>.json`. The manifest records your GPU, dtype,
library versions and per-checkpoint timings — reproducibility in the paper depends on it.
