# bio-contrast-unified

One configurable codebase for the Bio-Contrast PDX **and** PDO experiments, with the
result-integrity fixes applied. **No data is copied** — `data`, `pdo_data`, `KEGG`, `KGML`
are symlinks into the existing folders.

## What changed vs the original folders
| Area | Original | Here |
|---|---|---|
| CV splits | `StratifiedShuffleSplit(n_splits=cv)` → **overlapping** test folds | `splits.py`: `StratifiedKFold` → **disjoint** test folds + inner val split |
| Single-positive test folds | filtered/degenerate | **kept** (deemed legit; `min_pos=1`) |
| PDO expression | raw, uncorrected | `combat.py`: **ComBat** by dataset of origin (`spec.apply_combat`) |
| PDX family expansion | ON (replicate leakage) | **OFF** by default (`apply_family_expansion=False`) |
| PDX vs PDO | two divergent `data.py` | one `data.py` + `DatasetSpec` |
| Parallelism | N hand-copied folders, drug list reversed | `launch.py`: **N-GPU** scheduler, N auto-detected/flexible |

## Layout
```
dataset_spec.py       PDX/PDO specs + experiments.yaml loader
config/experiments.yaml   the (dataset x variant) matrix
splits.py             disjoint CV (the fix)          [unit-tested]
combat.py             ComBat batch correction        [unit-tested]
data.py               UnifiedDataloaderFactory (spec-driven; wires splits + combat)
run_experiment.py     per-worker: shards drugs, runs baselines or contrastive
launch.py             multi-GPU scheduler (flexible N)
model.py modules.py trainer.py utils.py ...   copied compute code (from bio-contrast-3)
_ref_data_pdx.py _ref_data_pdo.py             original data.py variants (provenance)
```

## Run everything in parallel
```bash
# use all visible GPUs (auto-detected)
python launch.py --out results_unified

# explicit 20 GPUs, 2 shards each (finer load balancing), longer training
python launch.py --num-gpus 20 --shards-per-gpu 2 --epochs 2000

# see the job matrix without launching
python launch.py --dry-run
```
GPU count is never hardcoded: it comes from `--gpus`, `--num-gpus`, `$CUDA_VISIBLE_DEVICES`,
or `torch.cuda.device_count()`, in that order.

## Tests (no GPU / no real data needed)
```bash
python tests/test_fixes.py
```
Checks that test folds are disjoint, single-positive folds survive, and ComBat removes
cross-dataset batch shift.

## Status
- `splits.py`, `combat.py`, `dataset_spec.py`, `launch.py` — complete + unit-tested.
- `data.py` / `run_experiment.py` — complete and structured; **need a GPU smoke test against
  the real files** to confirm PDO column names (`combat_batch_column`) and the trainer wiring.
  Set the real dataset-of-origin column in `config/experiments.yaml` if it is not `source`.
