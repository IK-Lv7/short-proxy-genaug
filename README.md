# short-proxy-genaug

Code, configurations and per-seed results for the paper "Can a Short Training Run Choose a Generative Data Augmentation?".

This repository contains only what is needed to check and reproduce the experiments and analyses in the paper.
Datasets, generated images and model weights are not included (see "Reproducing").

**Sign of G.** The paper defines G = (loss on real images) − (loss on generated images).
`results/all_conditions.csv`, `results/review/`, `results/review2/` and `results/review3/` store the opposite sign (generated − real); the column or predictor is labelled `G<B>` or `-G` there.
`results/review5/`, `results/review6/` and `results/food_summary.csv` use the paper's sign.
Every run file in `results/runs/` (after extracting) has `fit_loss_real` and `fit_loss_gen`, from which G can be computed with either sign.

## Contents

| Path | Content |
|---|---|
| `src/augpred/` | Library code: data, models, training, generation, metrics |
| `scripts/` | Command-line entry points (below) |
| `configs/` | Experiment settings (`exp/`: conditions and seeds) and generation settings (`gen/`) |
| `splits/` | Indices of the low-data training and validation subsets |
| `results/runs.tar.gz` | Per-seed result of every training run (one JSON file per run, with the configuration and the training curve). Extract it first: `tar -xzf results/runs.tar.gz -C results` |
| `results/runs.csv`, `results/conditions.csv` | Per-seed and per-condition summaries of the runs |
| `results/all_conditions.csv` | One row per condition: Δ, p-value, training-free metrics, P and G |
| `results/metrics.csv`, `results/metrics2.csv`, `results/ds_metrics.csv` | Training-free metrics |
| `environment.yml` | Software environment |

## Where each part of the paper comes from

| Paper | Configs | Scripts | Results |
|---|---|---|---|
| Table 1 (fixed iteration budget) | `exp/phase1_seed_variance.yaml`, `exp/pilot_gen.yaml`, `exp/fixedsteps_gen.yaml` | `run_exp.py`, `aggregate.py` | `runs.csv` |
| Generated images (Figure 1) | `gen/*.yaml` | `generate.py` | (not included) |
| Training-free metrics | | `compute_metrics.py`, `compute_metrics2.py`, `compute_ds.py` | `metrics*.csv`, `ds_metrics.csv` |
| P and G (proxy runs) | same experiment files as the full runs | `run_exp.py --steps <B> --seeds 0 1 2 --prefix proxyfit<B>__` | `runs/proxyfit*` |
| Table 2, Figure 2, Appendix D | | `build_table.py`, `review_analyses.py` | `all_conditions.csv`, `review/` |
| Table 3 (trivial predictors, early-training signals) | | `review2_baselines.py`, `review2_early.py` | `runs/cart*`, `runs/trunc*`, `review2/` |
| Table 4 (selection) | | `selection_sim.py` | `selection_sim.csv` |
| Table 5 (cost) | | `cost_table.py` (needs the generated images) | `cost.csv` |
| Table 6 (on top of TrivialAugment, CIFAR-100) | `exp/combo_gen.yaml` | `run_exp.py` | `runs/combo_gen`, `runs/proxyfit1000__combo_gen` |
| Table 7 (dropping / regenerating images) | `exp/filter_gen.yaml`, `gen/filter.yaml` | `run_exp.py --sample-loss-dir`, `make_filter_scores.py` | `runs/proxysample1000__*`, `filter_scores/`, `runs/filter_gen` |
| Table 8 (robustness) | | `review3_analyses.py` | `review3/` |
| Table 9 (Food-101) | `exp/food_*.yaml`, `gen/food.yaml` | `prepare_food101.py`, `food_analysis.py` | `runs/food_*`, `food_summary.csv` |
| Tables 10 and 11 (Flowers102 with TrivialAugment; near-duplicates and shortcuts) | `exp/flowers_ta_gen.yaml`, `exp/shortcut_gen.yaml` | `make_shortcut_sets.py`, `review5_analysis.py` | `review5/` |
| Table 12 (reduced diversity) | `exp/review6_gen.yaml`, `gen/review6.yaml` | `review6_diversity.py`, `review6_analysis.py` | `review6/` |
| Appendix B (training and generation details) | `base.yaml`, `exp/*.yaml`, `gen/*.yaml` | | |

The experiment files group conditions by setting: `fixedsteps_gen`, `labelnoise_gen`, `inpaint_gen`, `ratio_gen`, `heldout_gen`, `sdxl_gen` (CIFAR-100, ResNet-18), `sub1_gen` (second CIFAR-100 subset), `vgg_gen` (VGG-11), `tin_gen`, `tin_gen2` (Tiny-ImageNet), `c100pt_gen`, `vit_gen` (pretrained ResNet-50 and ViT-B/16 on CIFAR-100), `flowers_gen` (Flowers102), `food_r50_gen`, `food_vit_gen` (Food-101), `combo_gen`, `flowers_ta_gen` (on top of TrivialAugment), `filter_gen`, `shortcut_gen`, `review6_gen`, `metric_models` (classifiers used by some training-free metrics).
`*_phase1` and `steps<N>__` runs choose the iteration budget; `extra__` runs are additional seeds.

## Reproducing

There are two ways to use this repository.
(A) checks the numbers in the paper from the included results and needs no GPU or dataset.
(B) reruns the experiments from scratch and needs a GPU, the datasets and several days of compute.

### Setup (both)

```bash
conda env create -f environment.yml && conda activate augpred
tar -xzf results/runs.tar.gz -C results    # unpack the per-seed results (every analysis reads them)
```

### (A) Recompute the paper's numbers from the included results

Each script reads `results/` and prints or writes the numbers of the tables it is listed for in "Where each part of the paper comes from".
LaTeX tables that the scripts produce are written to `results/tables/`.

```bash
python scripts/aggregate.py               # per-seed results -> results/runs.csv, results/conditions.csv (Delta per condition)
python scripts/build_table.py             # all conditions with Delta, FID, DS, P, G -> results/all_conditions.csv (Table 2, Figure 2)
python scripts/review_analyses.py         # rank correlations without label noise, bootstrap intervals (Table 2, Appendix D)
python scripts/review2_baselines.py       # trivial predictors and partial correlation controlling for strength and ratio (Table 3)
python scripts/review2_early.py           # confidence, EL2N, learning-curve extrapolation and validation loss (Table 3)
python scripts/selection_sim.py           # simulation of choosing a candidate (Table 4)
python scripts/review3_analyses.py        # robustness: top half, NDCG, leave one family out (Table 8)
python scripts/food_analysis.py           # Food-101 (Table 9)
python scripts/review5_analysis.py        # Flowers102 with TrivialAugment, near-duplicates and shortcuts (Tables 10 and 11)
python scripts/review6_analysis.py        # sets with reduced diversity (Table 12)
```

### (B) Rerun the experiments

Steps 1 to 4 are in the order they must be run.

**1. Prepare the datasets** (downloads or reads the raw data and saves the low-data subsets given in `splits/`)

```bash
python scripts/prepare_cifar100.py        # also prepare_tinyimagenet.py, prepare_flowers102.py, prepare_food101.py
```

**2. Generate the images** (Stable Diffusion; the images are cached under `data/generated/` and are not regenerated)

```bash
python scripts/generate.py configs/gen/pilot.yaml
```

Each experiment file in `configs/exp/` needs the images of these generation files:

| Experiment files | Generation files |
|---|---|
| `fixedsteps_gen`, `labelnoise_gen`, `ratio_gen`, `pilot_gen` | `pilot.yaml` |
| `inpaint_gen`, `heldout_gen`, `sdxl_gen` | the file of the same name |
| `combo_gen`, `vgg_gen`, `c100pt_gen`, `vit_gen` | `pilot.yaml`, `inpaint.yaml`, `heldout.yaml` (`c100pt_gen` and `vit_gen` also `sdxl.yaml`) |
| `filter_gen` | `pilot.yaml`, `heldout.yaml`, `filter.yaml` |
| `sub1_gen`, `tin_gen`, `tin_gen2`, `flowers_gen`, `food_*_gen`, `review6_gen` | the file of the same name (`flowers.yaml`, `food.yaml`, ...; `tin_gen2` uses `tin.yaml`; `flowers_ta_gen` uses `flowers.yaml`) |
| `shortcut_gen` | `python scripts/make_shortcut_sets.py` (built from existing generated images, no diffusion model) |

**3. Train the full runs** (train with the images added and measure the test accuracy; this gives Delta; one condition x seed per result file)

```bash
python scripts/run_exp.py configs/exp/fixedsteps_gen.yaml
```

**4. Train the proxy runs** (the short runs that give P and G; 1000 iterations from scratch, 250 for the pretrained models)

```bash
python scripts/run_exp.py configs/exp/fixedsteps_gen.yaml --steps 1000 --seeds 0 1 2 --prefix proxyfit1000__
```

Extra runs for Table 3 (the early-training signals the paper compares G with):

```bash
python scripts/run_exp.py configs/exp/fixedsteps_gen.yaml --steps 1000 --seeds 0 1 2 --prefix cart1000__ --fit-checkpoints 5   # per-image outputs during training (confidence, EL2N)
python scripts/run_exp.py configs/exp/fixedsteps_gen.yaml --stop-at 1000 --seeds 0 1 2 --prefix trunc1000__                     # full schedule stopped early (learning-curve extrapolation)
```

Then run the scripts of (A).
Training-free metrics (FID, KID, Discriminative Span and others) are computed before that with
`python scripts/compute_metrics.py <exp.yaml>`, `compute_metrics2.py` and `compute_ds.py` (they write `results/metrics*.csv` and `results/ds_metrics.csv`).

Runs whose results already exist are skipped (`--force` overwrites, `--only <condition>` selects conditions).
Experiments were run on a single 16 GB GPU, one job at a time.
Tiny-ImageNet is subject to the ImageNet terms of access; images derived from it are not distributed.

## License

MIT (see `LICENSE`).
