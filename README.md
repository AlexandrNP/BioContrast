# BioContrast

Biologically-informed contrastive representation learning for transferring drug-response
predictors from cancer **cell lines** to data-scarce clinical and pre-clinical models —
patient-derived xenografts (**PDX**), patient-derived organoids (**PDO**), and patient
tumors (**TCGA**).

Cell-line screens provide abundant, cheap drug-response labels, but models trained on them
generalize poorly to patient-relevant systems because of large domain shift. BioContrast
learns a shared representation in which paired cell-line and target-domain samples are
aligned, so a response predictor trained mostly on cell-line labels transfers to the target
domain. Gene-expression features are restricted to, and optionally organized by, curated
**KEGG pathways**, which injects prior biological structure into the encoder.

## Method

- **Pathway-structured features.** Expression is subset to a gene set (KEGG by default) and
  can be routed through pathway/hierarchy-aware layers (`KEGGHierarchicalCNN`,
  `KEGGPathwayBottleneck`, graph-convolutional pathway filters) built from the bundled KEGG
  pathway definitions.
- **Two-domain contrastive alignment.** A `CellLineEncoder` and a target-domain encoder
  (`PDXEncoder`) feed a shared `ContrastiveProjection`; a CLIP-style contrastive loss pulls
  paired cell-line/target samples together while a response-prediction head is trained on
  the (label-rich) cell-line domain. Direct predictors provide baselines/ablations.
- **Dataset as configuration.** A single code path serves both targets; `dataset_spec.py`
  selects PDX vs PDO — file paths, response columns, ComBat batch correction, sample
  thresholds — as data rather than forked code.
- **Leakage-controlled evaluation.** Disjoint `StratifiedKFold` cross-validation with an
  inner validation split (`splits.py`); ComBat correction of target expression by dataset of
  origin for PDO (`combat.py`).

## Installation

```bash
conda env create -f environment.yml   # creates the 'bio-contrast' environment (CUDA 12.0)
conda activate bio-contrast
```

## Data setup

The repository ships the **code and the KEGG reference data**, not the expression/response
tables. Point the dataset specs at your copies (real files or symlinks) under the paths
declared in `dataset_spec.py`:

```
data/                                  # PDX_SPEC.data_root
  GeneSets/KEGG.txt                    # gene set used for features (also all.txt, lincs1000_list.txt, ...)
  Cell_Line_Drug_Screening_Data/       # cell-line expression, response, drug tables + PDX data
pdo_data/raw_data/                     # PDO_SPEC.data_root
  x_data/  y_data/                     # PDO expression, response, drug/metadata tables
KEGG/ , KGML/                          # KEGG pathway definitions (KEGG/ committed; refresh KGML with download.sh)
```

The exact file names for each dataset are listed in `PDX_SPEC` / `PDO_SPEC` in
`dataset_spec.py`; override any of them via `config/experiments.yaml` (e.g. the ComBat
`combat_batch_column`). To refresh the KEGG KGML pathway maps:

```bash
bash download.sh                       # fetches human (hsa) KGML into KGML/
```

## Usage

Train the transfer model for every drug that has paired cell-line/target splits, across CV
folds, via `run.py`:

```bash
# PDX target, KEGG features, device from config/config.yaml
python run.py --dataset pdx

# PDO target (applies ComBat correction)
python run.py --dataset pdo

# quick end-to-end smoke: one drug, two epochs, on CPU
python run.py --dataset pdx --limit-drugs 1 --steps 2 --steps-per-epoch 5 --device cpu
```

Options: `--dataset {pdx,pdo}`, `--gene-set {KEGG,ALL,LINCS,...}`, `--device`,
`--limit-drugs N`, `--steps N` (overrides `total_training_steps`), `--steps-per-epoch N`.
Per-drug/-fold checkpoints and best-model records are written under `log/`; a run resumes by
skipping folds whose `log/best-*.pickle` already exists.

## Configuration

`config/config.yaml` holds the model architecture (encoder/projection/predictor dimensions,
contrastive temperature, dropout), `total_training_steps`, and the default `device`.
`dataset_spec.py` (optionally overridden by `config/experiments.yaml`) holds everything that
differs between the PDX and PDO datasets.

## Repository layout

```
run.py            training entrypoint (dataset-selectable, argparse)
dataset_spec.py   PDX/PDO dataset specifications + experiments.yaml loader
data.py           spec-driven dataloader factory (wires splits + ComBat + gene sets)
splits.py         disjoint stratified K-fold cross-validation
combat.py         ComBat batch correction
model.py          encoders, contrastive projection, predictors, CellLineTransferLearner
modules.py        neural building blocks incl. KEGG pathway/hierarchy layers
trainer.py        transfer-learning training loop, checkpointing, evaluation
kegg.py hierarchies.py   KEGG pathway parsing and hierarchy construction
configuration.py logger.py utils.py rdconf.py   supporting utilities
config/config.yaml       model + training configuration
environment.yml          conda environment
download.sh              fetch KEGG KGML pathway maps
KEGG/ KGML/              KEGG pathway reference data
```

## License

See [LICENSE](LICENSE).
