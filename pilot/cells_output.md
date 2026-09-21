# 1

{
"python": "3.12.13",
"torch": "2.10.0+cu128",
"transformers": "5.0.0",
"cuda": true,
"gpu": "Tesla T4",
"vram_gb": 15.6
}
GPU: Tesla T4 | capability: sm_75
this torch was built for: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
GPU preflight: OK (fp16 matmul + embedding lookup)

on_kaggle: True
checkpoints -> /kaggle/temp/ckpt (1103 GB free)
results -> /kaggle/working
Internet: OK

# 2

240 facts x 5 checkpoints

# 3

shard 0: 1586 rows
shard 1: 1586 rows
shard 2: 1585 rows
shard 3: 1585 rows
shard 4: 1585 rows
shard 5: 1585 rows
shard 6: 1585 rows
shard 7: 1585 rows
shard 8: 1585 rows

14267 facts in 19s
Relations: 16 | all have templates

bucket
0 2450
1 2307
2 1586
3-4 1999
5-7 1556
8-13 1437
14-25 985
26-60 769
61-200 522
200+ 656

Control groups:
subject absent from corpus : 880
subject & object never co-occur: 2450

# 4

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
tokenizer ok | pad: 100277 | bos: 100257
240 probes built
gold token len 2.94 vs distractor 2.94 (should be close)

Example probe:
{
"fact_id": 309956,
"subj": "On the Run",
"prop": "director",
"obj": "Ernest Morris",
"bucket": "0",
"n_shared": 0,
"n_subject": 1,
"prompt": "The director of On the Run is",
"candidates": [
"Ernest Morris",
"Lal Jose",
"William Hanna",
"Christopher Guest",
"Jason Bloom",
"Edgar Wright",
"Adam Bernstein",
"Peter Horton",
"Ron Mann",
"Lance Comfort"
],
"possible_answers": [
"Ernest Morris"
]
}

# 5

scoring functions defined
2400 scored pairs + 1910 neutral + 240 generations per checkpoint
prior_idx == gold for 57 / 240 probes

# 5b

dtype calibration on step1000 (a trained checkpoint)
float16 {'finite': False, 'sec': 0.72, 'nonfinite': '128/128'}
bfloat16 {'finite': True, 'sec': 1.54, 'nonfinite': '0/128'}
float32 {'finite': True, 'sec': 1.02, 'nonfinite': '0/128'}

chosen dtype: float32 (fastest option with finite outputs)

# 6

step0: 70s (dl 20s, gpu 48s) | acc=0.096 conf=0.214
step1000: 57s (dl 0s, gpu 55s) | acc=0.221 conf=0.392
step3000: 90s (dl 31s, gpu 55s) | acc=0.263 conf=0.391
step6000: 88s (dl 30s, gpu 55s) | acc=0.271 conf=0.394
step10000: 89s (dl 31s, gpu 55s) | acc=0.275 conf=0.417

1200 rows, 5 checkpoints, all finite

# 7

step0 (random init) — correct confidence here is chance = 0.100
norm: acc=0.096 conf=0.214
sum: acc=0.092 conf=0.429
pmi: acc=0.125 conf=0.405
final: acc rare=0.125 vs frequent=0.604

[PASS] step0_acc_is_chance
[PASS] step0_conf_is_low
[PASS] norm_best_calibrated_at_init
[PASS] acc_rises_with_exposure
[PASS] generation_nonempty
[PASS] generation_varies
[PASS] all_values_finite
[PASS] all_checkpoints_present

# 8

Aggregate over training steps:

       acc_norm  conf_norm

step  
0 0.096 0.214
1000 0.221 0.392
3000 0.262 0.391
6000 0.271 0.394
10000 0.275 0.417

confidence at step0 : 0.214
confidence at step1000 : 0.392
confidence at plateau : 0.417
fraction of the rise already complete by step1000: 87.7%

VERDICT: SATURATED — onset is faster than public checkpoints can resolve. H2 (onset timing) closes as a negative result; pivot the paper to gap MAGNITUDE.

# 9

=== acc_norm ===
step 0 1000 3000 6000 10000
bucket  
0 0.083 0.208 0.042 0.125 0.042
1 0.042 0.208 0.250 0.167 0.167
2 0.125 0.125 0.250 0.167 0.167
3-4 0.125 0.125 0.208 0.167 0.208
5-7 0.083 0.125 0.042 0.083 0.125
8-13 0.167 0.208 0.250 0.250 0.333
14-25 0.125 0.208 0.375 0.250 0.208
26-60 0.042 0.292 0.333 0.333 0.292
61-200 0.042 0.250 0.417 0.500 0.542
200+ 0.125 0.458 0.458 0.667 0.667

=== conf_norm ===
step 0 1000 3000 6000 10000
bucket  
0 0.245 0.386 0.350 0.364 0.367
1 0.210 0.393 0.378 0.405 0.418
2 0.209 0.366 0.359 0.355 0.373
3-4 0.215 0.382 0.348 0.359 0.352
5-7 0.195 0.329 0.351 0.348 0.327
8-13 0.193 0.350 0.350 0.372 0.379
14-25 0.221 0.405 0.393 0.359 0.403
26-60 0.219 0.438 0.461 0.441 0.459
61-200 0.205 0.385 0.386 0.400 0.406
200+ 0.233 0.489 0.532 0.535 0.682

=== GAP (conf - acc) ===
step 0 1000 3000 6000 10000
bucket  
0 0.161 0.178 0.309 0.239 0.325
1 0.168 0.184 0.128 0.238 0.252
2 0.084 0.241 0.109 0.189 0.207
3-4 0.090 0.257 0.140 0.193 0.144
5-7 0.111 0.204 0.309 0.265 0.202
8-13 0.027 0.142 0.100 0.122 0.046
14-25 0.096 0.197 0.018 0.109 0.194
26-60 0.177 0.146 0.128 0.108 0.167
61-200 0.164 0.135 -0.031 -0.100 -0.135
200+ 0.108 0.031 0.074 -0.132 0.016

Prior-collapse rate among wrong answers: 0.317 (chance ~ 0.111)
bucket
0 0.209
1 0.299
2 0.380
3-4 0.329
5-7 0.425
8-13 0.380
14-25 0.352
26-60 0.288
61-200 0.218
200+ 0.214

## PNG IS IN THIS FOLDER (/PILOT)

# 10

Real run is 8.3x this pilot's probe volume.
Estimate per checkpoint at full scale: measure from the per-step times printed in Section 6,
multiply the scoring portion by the factor above (download time does not scale).

Example generations at the final pilot checkpoint:
[ 0] 'The director of On the Run is' -> 'the director of the Center for the Study of t' (gold='Ernest Morris', ok=0, conf=0.22)
[ 0] 'The author of Max is' -> 'a former member of the German Nazi Party, and' (gold='Barbro Lindgren', ok=0, conf=0.14)
[ 0] 'Sue Vertue was born in the city of' -> 'Paris, and was raised in the city of Paris' (gold='Surrey', ok=0, conf=0.22)
[ 0] 'The occupation of Nina Varlamova is' -> 'a major source of information about the activ' (gold='politician', ok=0, conf=0.17)
[ 0] 'The genre of Hamilton is' -> 'the "patriarchal" of the city' (gold='action film', ok=0, conf=0.11)
[ 0] 'The composer of Time is' -> 'a Man\n\nThe composer of Time is a' (gold='Jeff Daniels', ok=0, conf=0.34)
[ 0] 'The occupation of Howard Fowles is' -> 'a major source of information on Howard Fowle' (gold='politician', ok=0, conf=0.28)
[ 0] 'The occupation of Edward F. Cox is' -> 'described in the book "The Battle of the Wild' (gold='politician', ok=0, conf=0.18)

# FINAL REPORT:

========================================================================
PILOT REPORT — copy everything below back to Claude
========================================================================
{
"env": {
"python": "3.12.13",
"torch": "2.10.0+cu128",
"transformers": "5.0.0",
"cuda": true,
"gpu": "Tesla T4",
"vram_gb": 15.6,
"gpu_capability": "sm_75",
"torch_arch_list": [
"sm_70",
"sm_75",
"sm_80",
"sm_86",
"sm_90",
"sm_100",
"sm_120"
],
"scratch": "/kaggle/temp/ckpt",
"out": "/kaggle/working",
"free_gb": 1102.5
},
"config": {
"n_per_bucket": 24,
"n_cand": 10,
"model": "dhgottesman/LMEnt-1B-6E",
"steps": [
"step0",
"step1000",
"step3000",
"step6000",
"step10000"
]
},
"n_facts_total": 14267,
"bucket_counts": {
"0": 2450,
"1": 2307,
"2": 1586,
"3-4": 1999,
"5-7": 1556,
"8-13": 1437,
"14-25": 985,
"26-60": 769,
"61-200": 522,
"200+": 656
},
"n_probes": 240,
"len_match": {
"gold": 2.942,
"distractor": 2.936
},
"dtype_calibration": {
"float16": {
"finite": false,
"sec": 0.72,
"nonfinite": "128/128"
},
"bfloat16": {
"finite": true,
"sec": 1.54,
"nonfinite": "0/128"
},
"float32": {
"finite": true,
"sec": 1.02,
"nonfinite": "0/128"
}
},
"dtype_chosen": "float32",
"timing": {
"step0": {
"download_s": 19.6,
"gpu_s": 47.5,
"total_s": 70.1
},
"step1000": {
"download_s": 0.0,
"gpu_s": 55.0,
"total_s": 57.2
},
"step3000": {
"download_s": 31.4,
"gpu_s": 55.1,
"total_s": 89.5
},
"step6000": {
"download_s": 29.5,
"gpu_s": 55.2,
"total_s": 87.7
},
"step10000": {
"download_s": 31.0,
"gpu_s": 55.1,
"total_s": 89.1
}
},
"step0_scorers": {
"norm": {
"acc": 0.096,
"conf": 0.214
},
"sum": {
"acc": 0.092,
"conf": 0.429
},
"pmi": {
"acc": 0.125,
"conf": 0.405
}
},
"sanity": {
"step0_acc_is_chance": true,
"step0_conf_is_low": true,
"norm_best_calibrated_at_init": true,
"acc_rises_with_exposure": true,
"generation_nonempty": true,
"generation_varies": true,
"all_values_finite": true,
"all_checkpoints_present": true
},
"onset": {
"conf_step0": 0.214,
"conf_step1000": 0.392,
"conf_plateau": 0.417,
"frac_by_1000": 0.8768472906403942,
"verdict": "SATURATED \u2014 onset is faster than public checkpoints can resolve. H2 (onset timing) closes as a negative result; pivot the paper to gap MAGNITUDE."
},
"gap_final": {
"0": 0.325,
"1": 0.252,
"2": 0.207,
"3-4": 0.144,
"5-7": 0.202,
"8-13": 0.046,
"14-25": 0.194,
"26-60": 0.167,
"61-200": -0.135,
"200+": 0.016
},
"prior_collapse_overaדll": 0.3169705469845722,
"generated": "2026-08-28 15:13:09"
}
========================================================================

Files in /kaggle/working :
**notebook**.ipynb (0.27 MB)
pilot**LMEnt-1B-6E**step0.parquet (0.04 MB)
pilot**LMEnt-1B-6E**step1000.parquet (0.04 MB)
pilot**LMEnt-1B-6E**step10000.parquet (0.04 MB)
pilot**LMEnt-1B-6E**step3000.parquet (0.04 MB)
pilot**LMEnt-1B-6E**step6000.parquet (0.04 MB)
pilot_all.parquet (0.12 MB)
pilot_curves.png (0.21 MB)
pilot_report.json (0.00 MB)

Also send: pilot_curves.png, and the Section 8 VERDICT line.
