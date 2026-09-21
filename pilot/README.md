# Pilot — Tomer only

**Do not send this to Chen, Ofek, or Victoria yet.** The point of the pilot is to find bugs and
answer one open research question *before* three other people spend ~14 GPU-hours.

## What to do

1. Upload `00_pilot.ipynb` to Kaggle (**New Notebook -> File -> Import Notebook**).
2. **Settings -> Accelerator -> GPU P100.**
3. **Settings -> Internet -> On.** (Needs phone verification on your Kaggle account. This is the
   single most common failure; the notebook stops immediately with a clear message if it is off.)
4. **Run All.**
5. Wait ~25 minutes.
6. Copy back to me: the whole `PILOT REPORT` JSON block at the end, the `VERDICT` line from
   Section 8, and `pilot_curves.png`.

## Expected runtime

| Section | Time |
|---|---|
| 1-2 environment | <1 min |
| 3 load fact metadata (14,267 facts) | ~1 min |
| 4 build probes | ~1 min |
| 5b dtype calibration | ~2 min (one extra checkpoint load per dtype) |
| 6 checkpoint sweep, 5 x LMEnt-1B-6E | ~5-15 min depending on chosen dtype |
| 7-10 analysis | <1 min |

## Where files go (you do not need to configure this)

Kaggle gives you two writable places, and they behave differently:

| Path | What it is | Used for |
|---|---|---|
| `/kaggle/working` | Saved as your notebook output. **20 GB cap.** | Results only — small parquet files, a few MB total |
| `/kaggle/temp` | Scratch. Wiped when the session ends, does **not** count against the cap | The model checkpoints (5.3 GB each) |

The notebook picks these automatically and prints what it chose in Section 1. If `/kaggle/temp`
is unavailable it falls back to `/tmp`, then to `/kaggle/working` with a visible warning. It also
checks you have >12 GB free before starting and stops with a clear message if not.

Disk stays under ~6 GB at any moment: each checkpoint is downloaded, scored, then **deleted**
before the next one starts. Results are written after every checkpoint, so if the session dies
you can just re-run and it picks up where it stopped.

## Accelerator: use T4 x2, not P100

Recent PyTorch builds dropped CUDA kernels for Maxwell, Pascal, and Volta (`sm_50` / `sm_60` /
`sm_70`). Kaggle's **P100 is `sm_60`**, so it now crashes with
`CUDA error: no kernel image is available for execution on the device` as soon as a model runs.
This is an open, unresolved Kaggle bug ([docker-python #1546](https://github.com/Kaggle/docker-python/issues/1546)).

**T4 is `sm_75`, is supported, and is faster than P100 for this workload** — it has fp16 tensor
cores (~65 TFLOPS vs P100's ~19) and everything here runs in fp16. Choosing `GPU T4 x2` gives you
two T4s; the notebook uses one.

Section 1 now runs a GPU preflight (architecture check plus a real fp16 matmul and embedding
lookup). If the GPU is unusable it stops in about 3 seconds with the fix, instead of crashing
20 minutes into the sweep.

## Why you are re-running this

Your first run completed but produced **NaN for every trained checkpoint** — only `step0` gave
real numbers. Cause: LMEnt stores weights in fp32, and the 1B model's peak activation is
~3.06e6 (at `layers.12.mlp.down_proj`), which is **47x above fp16's 65504 ceiling**. Loading in
fp16 overflowed and every logit became NaN. A randomly initialised `step0` has tiny weights and
does not overflow, which is exactly why it looked like it worked.

Two fixes:

1. **Section 5b picks the dtype empirically**, calibrating on a *trained* checkpoint (never
   `step0`). It tries fp16, bf16, and fp32, verifies the outputs are finite, times each, and
   uses the fastest one that works. It prints the comparison so we can see it.
2. **Hard guards.** Any non-finite score now aborts the run at that checkpoint with a clear
   message, the sanity block checks for NaN and for degenerate (identical) generations, and
   Section 8 refuses to print a verdict if the curve has holes. The old version reported a
   confident `PARTIAL` verdict from entirely NaN data — that can no longer happen.

## What it is testing

**Bugs.** It exercises every stage the real runs use — metadata loading, probe construction,
three scoring functions, free generation, the mechanism probe, incremental save/resume.

**One research question.** On LMEnt-170M-6E, confidence goes `0.167` at step0 to `0.471` at
step10000 and is then flat for 648,000 steps. All the action is in the first 10k steps, and
`LMEnt-1B-6E` is the only released model with checkpoints in that window. So the pilot measures
step0 / 1000 / 3000 / 6000 / 10000 and reports what fraction of the confidence rise is already
done by step1000:

| Verdict | Meaning | What we do |
|---|---|---|
| `RESOLVABLE` | onset unfolds over step1000-10000 | H2 measurable; dense early window is the centrepiece |
| `PARTIAL` | mostly early but not instant | Usable; add step2000/4000/5000/7000/8000 |
| `SATURATED` | already done by step1000 | H2 becomes a reported negative result; paper pivots to gap magnitude |

All three are publishable. This just tells us which paper we are writing.

## Checklist before you send results back

- [ ] All **8** checks in Section 7 print `PASS`
- [ ] Section 5b printed a dtype comparison and chose one
- [ ] Section 8 printed a `VERDICT` line
- [ ] `pilot_report.json` exists in `/kaggle/working`
- [ ] `pilot_curves.png` exists and the three panels are not empty
- [ ] Copied the full `PILOT REPORT` JSON block
- [ ] Noted the per-checkpoint wall times printed in Section 6 (used to size everyone's runs)

If a check says `FAIL`, send the report anyway — that is exactly what the pilot is for.
