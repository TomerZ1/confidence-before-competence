# Confidence Before Competence

Entity frequency and calibration during language model pretraining.

Final project for NLP 2526b, Tel Aviv University. Tomer Zalberg, Chen Mizrahi, Ofek Tovli,
Victoria Shabanov.

**Paper:** [`paper/acl_latex.pdf`](paper/acl_latex.pdf)

## What this is

We score 2,993 PopQA facts as ten-way multiple choice across 61 checkpoints of four
[LMEnt](https://doi.org/10.1162/tacl.a.746) models, and compare the model's confidence against its
accuracy as a function of how often the subject and answer entities co-occurred in the pretraining
corpus. A randomly initialised model already reports a confidence-accuracy gap of about 0.13.
Training closes that gap in proportion to exposure, and leaves it untouched for facts the corpus
has no evidence for.

## Reproducing

Everything runs on free Kaggle GPUs. Total cost was about 8 GPU-hours.

| Order | Notebook | Where | Time |
|---|---|---|---|
| 0 | `pilot/00_pilot.ipynb` | Kaggle, GPU | ~10 min |
| 1 | `01_data_prep/01_data_prep.ipynb` | CPU only | ~25 min |
| 2 | `runs/<person>/02_run_sweep.ipynb` | Kaggle, GPU, x4 in parallel | ~2 h each |
| 3 | `analysis/03_analysis.ipynb` | local, CPU | ~1 min |
| 4 | `analysis/04_baselines.ipynb` | local, CPU | seconds |

Step 1 is already done: its outputs are in `01_data_prep/output/`, so you can skip to step 2 or
straight to step 3 using the results in `runs/*/results/`.

### Two things that will break it

- **Accelerator must be `GPU T4 x2`, not P100.** Current PyTorch ships no kernels for sm_60, so
  P100 fails with `no kernel image is available`
  ([Kaggle/docker-python#1546](https://github.com/Kaggle/docker-python/issues/1546)).
- **Inference must run in float32.** LMEnt stores fp32 weights whose activations reach about
  3e6, far past the float16 ceiling. In float16 every trained checkpoint returns NaN while
  `step0` looks fine, so the failure is silent. The notebooks pin float32 and abort on any
  non-finite score.

## Layout

```
paper/              LaTeX source, bibliography, figures, compiled PDF
pilot/              end-to-end smoke test at small scale
01_data_prep/       builds the shared probe set; outputs in output/
runs/<person>/      the checkpoint sweep, one folder per runner
runs/<person>/results/   67 per-fact result files plus a manifest each
analysis/           figures and tables for the paper
```

The four runner notebooks are identical except for one line, `PERSON`, which selects a slice of
the checkpoint list. That is deliberate: the scoring code has to be byte-identical for the four
result sets to be poolable.

## Data

| What | Where |
|---|---|
| Probe set, exposure histograms, fact metadata | `01_data_prep/output/` in this repo, and as the Kaggle dataset `lment-probes` |
| Models | [`dhgottesman/LMEnt-*`](https://huggingface.co/collections/dhgottesman/lment-68a9dd370e1f746cacd8ce58) on HuggingFace |
| PopQA with corpus counts | [`dhgottesman/popqa-kas`](https://huggingface.co/datasets/dhgottesman/popqa-kas) |

The probe set is identified by the hash `5132e8b6ffaa40cb`. Every run recomputes it and refuses to
score if it differs, so all four result sets are known to cover the same 2,993 facts with the same
candidate options.

Each `manifest__<person>.json` records the GPU, numeric precision, torch and transformers
versions, and per-checkpoint timings for that runner.

## Results files

`runs/<person>/results/res__<person>__<model>__<templates>__step<N>.parquet`, one row per fact:
accuracy, confidence and gold probability under three scoring functions, the free-generation
output and its confidence, and the corpus-frequency and model-preference priors for the relation.
Every number in the paper derives from these.

## Note

`analysis/03_analysis.ipynb` and `04_baselines.ipynb` regenerate the figures and the main tables.
A few values quoted in the paper, including the initialisation-relative changes and the bootstrap
intervals on the AUCs, were computed separately from the same result files and are not reproduced
by those notebooks.
