"""
Unified data layer for Bio-Contrast (PDX + PDO in ONE module).

The two original `data.py` files diverged only in *data wiring*; here that wiring is a
`DatasetSpec` (see dataset_spec.py) and the code path is shared. Two integrity fixes are
baked in:

  1. Disjoint CV — cross-validation splits come from `splits.make_cv_splits`
     (StratifiedKFold outer folds), replacing the original overlapping
     StratifiedShuffleSplit. Single-positive test folds are kept on purpose.
  2. ComBat — for PDO, target-domain expression is batch-corrected by dataset of origin
     before any split (unsupervised; no label leakage).  [spec.apply_combat]

Family→sample expansion (the PDX replicate-leakage source) is OFF unless a spec explicitly
turns it on.

Loader method names match what trainer.py / run_experiment.py expect.
"""

import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder

from utils import binarize_auc_response, get_balanced_class_weights
from splits import make_cv_splits
from combat import combat

pickle.HIGHEST_PROTOCOL = 4

ID_COLUMNS = ['Sample', 'UniqueID']

# --------------------------------------------------------------------------- #
# Compatibility symbols: the copied compute modules (model/KEGG path) import   #
# these enums from `data`. Kept here so the unified data.py is a drop-in.       #
# --------------------------------------------------------------------------- #
import os as _os
from enum import Enum as _Enum
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_GENESETS = _os.path.join(_HERE, 'data', 'GeneSets')

# Path constants imported by the copied KEGG modules (kegg.py etc.).
DATA_DIR = _os.path.join(_HERE, 'data')
KEGG_DIR = _os.path.join(_HERE, 'KEGG')
KEGGID_DIR = _os.path.join(KEGG_DIR, 'KEGGID')
KEGG_GENE_DIR = _os.path.join(KEGG_DIR, 'Symbol')
PROCESSED_DIR = _os.path.join(DATA_DIR, 'Processed')


class Source(_Enum):
    ALL = 'ALL'; CCLE = 'CCLE'; CTRP = 'CTRP'; GDSC = 'GDSC'; gCSI = 'gCSI'; NCI60 = 'NCI60'


class GeneSet(_Enum):
    ALL = _os.path.join(_GENESETS, 'all.txt')
    LINCS = _os.path.join(_GENESETS, 'lincs1000_list.txt')
    ONCOGENES_REG = _os.path.join(_GENESETS, 'oncogenes_list.txt')
    ONCOGENES_DOCKING = _os.path.join(_GENESETS, 'oncogenes_gausschem4.txt')
    KEGG = _os.path.join(_GENESETS, 'KEGG.txt')


class DrugInfo(_Enum):
    SMILES = 'SMILES'; DESCRIPTORS = 'descriptors'; ECFP = 'ECFP'; PFP = 'PFP'


# KEGG pathway helpers (imported by the model's KEGG-CNN path). Loaded from the existing
# KEGG.pickle on disk — no network call.
KEGG_FILE = _os.path.join(_HERE, 'data', 'KEGG.pickle')


def get_kegg(organism='hsa'):
    if _os.path.isfile(KEGG_FILE):
        return pickle.load(open(KEGG_FILE, 'rb'))
    from Bio.KEGG.REST import kegg_list, kegg_get  # only if the cache is missing
    ref = [p.split('\t')[0].split('map')[1] for p in kegg_list('path').read().split('\n')
           if len(p) > 0 and 'map' in p]
    org = []
    for code in ref:
        try:
            org.append(kegg_get(f'{organism}{code}').read())
        except Exception:
            continue
    pickle.dump(org, open(KEGG_FILE, 'wb'))
    return org


def get_genes(pathways):
    processed_pathways, pathway_names, keggId2symbol = {}, {}, {}
    for pathway in pathways:
        is_gene = False
        pathway_idx = None
        pathway_name = None
        for line in pathway.split('\n'):
            if 'PATHWAY_MAP' in line:
                pathway_idx = line.split(' ')[1]
                pathway_name = line.split(pathway_idx)[-1].strip(' ')
                processed_pathways[pathway_name] = []
                pathway_names[pathway_name] = pathway_idx
            if ''.join(list(line)[:4]) == 'GENE':
                is_gene = True
            if 'COMPOUND' in line:
                break
            if is_gene:
                tokens = line.split(';')[0].split(' ')
                if len(tokens) < 3:
                    continue
                keggId2symbol[tokens[-3]] = tokens[-1]
                processed_pathways[pathway_name].append(tokens[-1])
    return processed_pathways, pathway_names, keggId2symbol


def get_pathways():
    return get_genes(get_kegg())


def fetch_kegg_pathway_hierarchy():
    _, pathway_codes = get_pathways()
    hierarchy = {}
    for pathway_name, pathway_id in pathway_codes.items():
        hierarchy.setdefault(pathway_name.split(" - ")[0], []).append((pathway_id, pathway_name))
    return hierarchy


# --------------------------------------------------------------------------- #
# Low-level loaders (spec-driven)                                             #
# --------------------------------------------------------------------------- #
def _norm_sample(series):
    """Canonicalise sample-id separators so response and expression tables join."""
    return (series.astype(str)
            .str.replace('~', '-', regex=False)
            .str.replace('.', '-', regex=False))


def get_gene_list(gene_set):
    # Accept either a path string (unified callers) or a GeneSet enum (copied KEGG modules).
    path = gene_set.value if hasattr(gene_set, 'value') else gene_set
    genes = list(pd.read_csv(path, header=None).transpose().values[0])
    return genes + ['Sample']


def load_expression(path, gene_set_file, sep='\t'):
    """Read expression restricted to the gene list, reindexed to the FULL list order.

    Genes present in the list but absent from the file are filled with 0, so the column
    positions line up 1:1 (same count, same order) with the KEGG-CNN hierarchy mapping,
    which indexes genes by position. Without this, a missing gene shifts every downstream
    column and the model's gene indices run off the end of the input (CUDA index assert).
    """
    if gene_set_file is None:
        return pd.read_csv(path, sep=sep)
    gene_list = get_gene_list(gene_set_file)                 # includes 'Sample'
    data = pd.read_csv(path, sep=sep, usecols=lambda x: x in gene_list)
    genes_only = [g for g in gene_list if g != 'Sample']
    out = {}
    if 'Sample' in data.columns:
        out['Sample'] = data['Sample'].values
    n = len(data)
    for g in genes_only:
        out[g] = data[g].values if g in data.columns else np.zeros(n)
    cols = (['Sample'] if 'Sample' in out else []) + genes_only
    df = pd.DataFrame(out, columns=cols)
    # Fill missing expression values with 0 (as the original loaders did). Un-filled NaNs
    # propagate through the encoder into NaN predictions and crash the metrics.
    df[genes_only] = df[genes_only].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df


class UnifiedDataloaderFactory:
    """
    One factory, parameterized by a DatasetSpec, that yields the drug-specific
    cell-line / target dataloaders and the paired (contrastive) dataloaders.
    """

    def __init__(self, spec, gene_set_file, device=None):
        self.spec = spec
        self.device = device
        self.gene_set_file = gene_set_file

        # ---- expression ----
        self.cell_line_rna = load_expression(spec.resolve('expression_file'), gene_set_file)
        self.target_rna = load_expression(spec.resolve('pdx_expression_file'), gene_set_file)
        if spec.kind == 'pdx' and 'Sample' in self.target_rna.columns:
            self.target_rna['Sample'] = _norm_sample(self.target_rna['Sample'])

        # ---- responses (normalized to Sample / UniqueID / Response) ----
        self.cell_line_response = self._load_cell_line_response()
        self.target_response, target_batch = self._load_target_response()

        # ---- ComBat on target expression, keyed by dataset of origin (PDO) ----
        if spec.apply_combat:
            self.target_rna = self._combat_correct(self.target_rna, target_batch)

        # ---- index everything on Sample ----
        self.cell_line_rna = self._index_expression(self.cell_line_rna)
        self.target_rna = self._index_expression(self.target_rna)

        # Keep only responses whose sample has expression (a response without expression
        # is unusable, and this prevents LabelEncoder 'unseen label' crashes).
        self.cell_line_response = self.cell_line_response[
            self.cell_line_response['Sample'].isin(self.cell_line_rna.index)].reset_index(drop=True)
        self.target_response = self.target_response[
            self.target_response['Sample'].isin(self.target_rna.index)].reset_index(drop=True)

        # binarized cell-line response for stratification
        self.binarized_cell_line_response = self.cell_line_response.copy()
        self.binarized_cell_line_response['Response'] = binarize_auc_response(
            self.cell_line_response['Response'])

        # label-encode sample ids to integers (RNA index + response Sample col)
        self.cl_enc = LabelEncoder()
        self.tg_enc = LabelEncoder()
        self.cell_line_rna.index = self.cl_enc.fit_transform(self.cell_line_rna.index)
        self.target_rna.index = self.tg_enc.fit_transform(self.target_rna.index)

        # A2: upload both (small) expression matrices to GPU ONCE here; every drug/fold
        # dataset then gathers rows from these shared tensors instead of re-copying.
        self._cl_preload = _preload_matrix(self.cell_line_rna, self.device)
        self._tg_preload = _preload_matrix(self.target_rna, self.device)
        self.cell_line_response['Sample'] = self.cl_enc.transform(self.cell_line_response['Sample'])
        self.binarized_cell_line_response['Sample'] = self.cl_enc.transform(
            self.binarized_cell_line_response['Sample'])
        self.target_response['Sample'] = self.tg_enc.transform(self.target_response['Sample'])

        self.cell_line_response.set_index('UniqueID', inplace=True)
        self.binarized_cell_line_response.set_index('UniqueID', inplace=True)
        self.target_response.set_index('UniqueID', inplace=True)

        self.cross_validation_num = spec.cv_folds
        self.validation_size = spec.val_size
        # BC_SEED overrides the CV/split random seed so the same drug can be run at multiple
        # independent stratified-shuffle-split seeds -> a real multi-seed CI on small-n drugs
        # (single-seed per-drug AUROC swings wildly at ~5 responders). Default = spec seed.
        self.random_seed = int(os.environ.get('BC_SEED', spec.random_seed))

        self.cell_line_splits = {}
        self.target_splits = {}
        self.skipped = []
        self._prepare_splits()

    # ---------- response loaders ----------
    def _load_cell_line_response(self):
        spec = self.spec
        df = pd.read_csv(spec.resolve('response_file'), sep='\t', low_memory=False)
        df = df[[spec.resp_sample_col, spec.resp_drug_col, spec.resp_value_col]]
        df.columns = ['Sample', 'UniqueID', 'Response']
        return df

    def _load_target_response(self):
        """Returns (response_df[Sample,UniqueID,Response], batch_series_aligned_to_expr)."""
        spec = self.spec
        raw = pd.read_csv(spec.resolve('target_response_file'), sep='\t', low_memory=False)

        if spec.kind == 'pdo':
            df = pd.DataFrame({
                'Sample': raw['Organoid'].astype(str),
                'UniqueID': raw['Drug'],
                'Response': (raw['AUC'] < 0.5).astype(int),
                # continuous responsiveness rho in [0,1] (1 = most responsive = low AUC), for the
                # negative-manifold geometry loss (BC_LOSS_TYPE=geo). Binary Response is unchanged.
                'ResponseCont': (1.0 - raw['AUC'].clip(0.0, 1.0)).astype(float),
            })
            df['Group'] = df['Sample']          # organoids: one sample == one group
            if os.environ.get('BC_GROUP_BY_DATASET') == '1' and 'Dataset' in raw.columns:
                # COHORT-BLOCKED CV (iter116): group by dataset-of-origin so StratifiedGroupKFold holds out ENTIRE
                # cohorts -> tests whether transfer survives to an UNSEEN cohort (controls the dataset confound, Part VII).
                df['Group'] = raw['Dataset'].astype(str).values
        else:  # pdx
            df = raw[['Sample', 'Drug', 'Response']].copy()
            df.columns = ['Sample', 'UniqueID', 'Response']
            df = self._map_pdx_drug_ids(df)                     # NSC id -> cell-line UniqueID
            df['Sample'] = _norm_sample(df['Sample'])           # patient/family-level id
            # PDX raw Response is a ranked responsiveness (higher = more responsive); keep it as
            # the continuous rho in [0,1] before binarising at >0.5 (orientation matches PDO).
            df['ResponseCont'] = df['Response'].astype(float).clip(0.0, 1.0)
            df['Response'] = (df['Response'] > 0.5).astype(int)
            # Response is family-level (e.g. NCIPDM-287954-098-R); expression is per-aliquot
            # (NCIPDM-112475-105-R-<aliquot>). Map each family response onto its individual
            # expression aliquots for the join, and keep the family as the split GROUP so
            # replicates of one tumor never straddle train/test (leak-free).
            df = self._expand_family(df)

        # batch (dataset of origin) for ComBat, aligned to target-expression samples
        batch = None
        if spec.apply_combat:
            batch = self._target_batch_labels()
        return df, batch

    def _map_pdx_drug_ids(self, df):
        """Map PDX NSC drug ids -> cell-line UniqueID so contrastive pairing can match drugs
        (port of the original map_nci_drug_id). Rows with no mapping are dropped."""
        try:
            info = pd.read_csv(self.spec.resolve('drug_map_file'), sep='\t', low_memory=False)
        except Exception as e:
            print(f"[pdx-drugmap] could not read drug map ({e}); leaving ids unmapped")
            return df
        nsc_cols = [c for c in ['NSC', 'NSC.ID(NCI_IOA_AOA_drugs)', 'NSC.ID(NCI60_drug)']
                    if c in info.columns]
        if 'UniqueID' not in info.columns or not nsc_cols:
            return df
        nsc2uid = {}
        for _, row in info[['UniqueID'] + nsc_cols].iterrows():
            for c in nsc_cols:
                nsc = str(row[c]).split('.')[-1]
                nsc2uid[nsc] = row['UniqueID']
        keep = df['UniqueID'].apply(lambda x: str(x).split('.')[-1] in nsc2uid)
        n_before = df['UniqueID'].nunique()
        df = df[keep].copy()
        df['UniqueID'] = [nsc2uid[str(x).split('.')[-1]] for x in df['UniqueID']]
        print(f"[pdx-drugmap] mapped {df['UniqueID'].nunique()}/{n_before} drugs to cell-line ids")
        return df

    def _expand_family(self, df):
        """Map family-level PDX response onto individual expression aliquots; Group=family."""
        indiv = self.target_rna['Sample'].astype(str)
        fam = indiv.map(lambda s: s.rsplit('-', 1)[0])
        fam_df = pd.DataFrame({'Sample_ind': indiv.values, 'Family': fam.values})
        merged = df.merge(fam_df, left_on='Sample', right_on='Family', how='inner')
        out = pd.DataFrame({
            'Sample': merged['Sample_ind'].values,
            'UniqueID': merged['UniqueID'].values,
            'Response': merged['Response'].values,
            'Group': merged['Family'].values,     # split group = family (leak-free)
        })
        if 'ResponseCont' in merged.columns:      # carry continuous responsiveness for the geo loss
            out['ResponseCont'] = merged['ResponseCont'].values
        return out

    def _target_batch_labels(self):
        """Dataset-of-origin per target sample, for ComBat. Read from metadata/source column.

        BUG-3 FIX (env-gated, reversible via BC_FIX_COMBAT=1; default OFF = original behavior):
        The default path below reads `combat_batch_column` from the SOURCE response/metadata
        table, which is keyed by cell-line ids. The organoid TARGET samples are absent from
        that table, so `_combat_correct`'s index intersection hits 0 organoids and ComBat
        ends up "correcting" the cell lines instead (a no-op for the target domain). With
        BC_FIX_COMBAT=1 (PDO only) we build a per-organoid Dataset-of-origin label directly
        from the target response file (PDO_response_combined_v2.tsv), keyed by the SAME sample
        id target_rna is indexed by (the 'Organoid' id == expression 'Sample'), so ComBat
        actually corrects the organoids across the PDO studies.
        """
        spec = self.spec
        if os.environ.get('BC_FIX_COMBAT') == '1' and spec.kind == 'pdo':
            raw = pd.read_csv(spec.resolve('target_response_file'), sep='\t', low_memory=False)
            bs = (raw[['Organoid', 'Dataset']].astype(str)
                  .drop_duplicates(subset=['Organoid'])
                  .set_index('Organoid')['Dataset'])
            print(f"[combat][BC_FIX_COMBAT=1] target batch labels from "
                  f"{os.path.basename(spec.resolve('target_response_file'))}: "
                  f"{bs.nunique()} datasets over {len(bs)} organoids "
                  f"-> {sorted(bs.unique().tolist())}")
            return bs
        col = spec.combat_batch_column
        # Try metadata file first, then the response/source table.
        for path in (spec.resolve('metadata_file'), spec.resolve('response_file')):
            if os.path.exists(path):
                meta = pd.read_csv(path, sep='\t', low_memory=False)
                sample_col = 'Sample' if 'Sample' in meta.columns else (
                    'sample_name' if 'sample_name' in meta.columns else spec.resp_sample_col)
                if col in meta.columns and sample_col in meta.columns:
                    return meta.set_index(sample_col)[col]
        return None

    def _combat_correct(self, expr, batch_series):
        """Apply ComBat to a samples x genes expression frame using dataset-of-origin batches."""
        if batch_series is None:
            print("[combat] no batch column found; skipping ComBat")
            return expr
        e = expr.copy()
        sample_col = 'Sample' if 'Sample' in e.columns else None
        if sample_col is None:
            return expr
        # Dedupe BOTH sides by sample id so expression rows and batch labels align 1:1.
        e = e.drop_duplicates(subset=[sample_col]).set_index(sample_col, drop=False)
        bs = batch_series[~batch_series.index.duplicated(keep='first')]
        gene_cols = [c for c in e.columns if c != 'Sample']
        common = e.index.intersection(bs.index)
        if len(common) < 3:
            print("[combat] <3 samples with known batch; skipping ComBat")
            return expr
        sub = e.loc[common, gene_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        batches = bs.loc[common].astype(str).values           # len == sub.shape[0]
        assert len(batches) == sub.shape[0], (len(batches), sub.shape)
        if len(pd.unique(batches)) < 2:
            print("[combat] single batch; nothing to correct")
            return expr
        print(f"[combat] correcting {sub.shape[0]} samples x {sub.shape[1]} genes "
              f"across {len(pd.unique(batches))} datasets of origin")
        corrected = combat(sub, batches)
        e.loc[common, gene_cols] = corrected.values
        return e.reset_index(drop=True)

    @staticmethod
    def _index_expression(expr):
        if 'Sample' in expr.columns:
            expr = expr.set_index('Sample')
        expr = expr[~expr.index.duplicated(keep='first')]
        return expr

    # ---------- disjoint CV splits (the fix) ----------
    def _prepare_splits(self):
        spec = self.spec
        drug_ids = np.unique(np.concatenate([
            self.cell_line_response.index.unique().values,
            self.target_response.index.unique().values,
        ]))
        for drug_id in drug_ids:
            # cell-line splits (stratified on binarized AUC)
            if drug_id in self.cell_line_response.index:
                cl = self.cell_line_response.loc[[drug_id]].reset_index()
                bcl = self.binarized_cell_line_response.loc[[drug_id]].reset_index()
                if bcl['Response'].sum() >= 2:
                    s = make_cv_splits(cl, bcl['Response'].values,
                                       spec.cv_folds, spec.val_size, self.random_seed)
                    if s is not None:
                        self.cell_line_splits[drug_id] = s

            # target (PDX/PDO) splits — keep single-positive test folds
            if drug_id in self.target_response.index:
                tg = self.target_response.loc[[drug_id]].reset_index()
                pos = int(tg['Response'].sum())
                neg = int(len(tg) - pos)
                if pos >= spec.min_pos and neg >= spec.min_neg:
                    grp = tg['Group'].values if 'Group' in tg.columns else None
                    s = make_cv_splits(tg, tg['Response'].astype(int).values,
                                       spec.cv_folds, spec.val_size, self.random_seed,
                                       groups=grp)
                    if s is not None:
                        self.target_splits[drug_id] = s
                    else:
                        self.skipped.append((drug_id, pos, neg))

    def paired_keys(self):
        """Drugs that have BOTH cell-line and target splits -> {drug: {cv_idx: {}}}."""
        keys = {}
        for drug_id in self.cell_line_splits:
            if drug_id in self.target_splits:
                folds = set(self.cell_line_splits[drug_id]) & set(self.target_splits[drug_id])
                keys[drug_id] = {cv: {} for cv in sorted(folds)}
        return keys

    # ---------- dataloaders ----------
    def _response_loaders(self, splits, rna_df, batch_size, num_samples=None, preloaded=None):
        loaders = {}
        for key, sub in splits.items():
            ds = ResponseDataset(sub, rna_df, self.device, preloaded=preloaded)
            if key == 'train' and num_samples is not None:
                weights = torch.ones(len(sub))
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples)
                loaders[key] = DataLoader(ds, batch_size=batch_size, sampler=sampler)
            else:
                loaders[key] = DataLoader(ds, batch_size=batch_size,
                                          shuffle=(key == 'train'))
        return loaders['train'], loaders['val'], loaders['test']

    def get_drug_specific_cell_line_dataloaders(self, drug_id, cv_idx, batch_size=128, num_samples=None):
        split = self.cell_line_splits[drug_id][cv_idx]
        samples = pd.concat([split[k]['Sample'] for k in split])
        rna = self.cell_line_rna.loc[samples]
        rna = rna[~rna.index.duplicated(keep='first')]
        return self._response_loaders(split, rna, batch_size, num_samples,
                                      preloaded=self._cl_preload)

    def get_drug_specific_pdx_dataloaders(self, drug_id, cv_idx, batch_size=128, num_samples=None):
        split = self.target_splits[drug_id][cv_idx]
        samples = pd.concat([split[k]['Sample'] for k in split])
        rna = self.target_rna.loc[samples]
        rna = rna[~rna.index.duplicated(keep='first')]
        return self._response_loaders(split, rna, batch_size, num_samples,
                                      preloaded=self._tg_preload)

    def get_paired_cell_line_pdx_loaders(self, drug_id, cv_idx, num_samples, batch_size=128):
        cl_split = deepcopy(self.cell_line_splits[drug_id][cv_idx])
        tg_split = self.target_splits[drug_id][cv_idx]
        out = {}
        for set_name in cl_split:
            cl_samples = np.unique(cl_split[set_name]['Sample'])
            tg_samples = np.unique(tg_split[set_name]['Sample'])
            # continuous cell-line responsiveness rho in [0,1] (1 = responsive = low AUC) for the
            # geo loss, computed BEFORE the AUC column is binarised in-place below.
            cl_split[set_name]['ResponseCont'] = (
                1.0 - pd.to_numeric(cl_split[set_name]['Response'], errors='coerce').clip(0.0, 1.0)).fillna(0.0)
            cl_split[set_name]['Response'] = binarize_auc_response(cl_split[set_name]['Response'])
            eff = num_samples if set_name == 'train' else 1000
            paired = DrugSpecificPairedDataset(
                eff, batch_size,
                self.cell_line_rna.loc[cl_samples], cl_split[set_name],
                self.target_rna.loc[tg_samples], tg_split[set_name],
                device=self.device,
                cl_preloaded=self._cl_preload, tg_preloaded=self._tg_preload)
            out[set_name] = DataLoader(paired, batch_size=1)
        return out['train'], out['val'], out['test']


# --------------------------------------------------------------------------- #
# Datasets (pandas-based; identical semantics to the original PDX versions)    #
# --------------------------------------------------------------------------- #
def _preload_matrix(rna_df, device):
    """Upload a full (Sample-indexed, gene-column) expression matrix to GPU ONCE.

    Returns (padded_tensor, {sample: row}). The tensor has a trailing all-zero row so a
    sample missing from the matrix gathers zeros -- matching the reindex(...)->nan_to_num
    semantics of the pandas path. ResponseDataset then indexes this shared GPU tensor
    instead of re-doing reindex->to_numpy->to(device) for every drug/fold (amend A2)."""
    df = rna_df[~rna_df.index.duplicated(keep='first')]
    mat = np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(mat).float()
    if device is not None:
        t = t.to(device)
    zero_row = torch.zeros(1, t.shape[1], dtype=t.dtype, device=t.device)
    padded = torch.cat([t, zero_row], dim=0)          # index len(df) == the zero row
    row_index = {s: i for i, s in enumerate(df.index)}
    return padded, row_index


class ResponseDataset(Dataset):
    """RNA/response dataset, materialized once into tensors.

    The original per-item `rna_df.loc[sample_id]` pandas lookup is O(n) and, for
    high-sample drugs (common drugs screened on thousands of cell lines), stalls the
    paired-dataset construction for many minutes. Here the RNA rows for every response are
    resolved ONCE with a vectorized `reindex` and cached as a tensor, so __getitem__ is an
    O(1) tensor index. This is the "cache the paired dataset" fix.
    """

    def __init__(self, response_df, rna_df, device, preloaded=None, with_cont=False):
        super().__init__()
        response_df = response_df.reset_index(drop=True)
        samples = response_df['Sample'].values
        resp = np.asarray(response_df['Response'].values, dtype=np.float32)
        # with_cont: also carry a continuous responsiveness channel (rho in [0,1]) so the
        # paired dataset can build the negative-manifold geometry (BC_LOSS_TYPE=geo). resp
        # becomes a 2-col tensor [:,0]=binary Response, [:,1]=ResponseCont. Falls back to the
        # binary value if ResponseCont is absent (so non-geo runs are byte-identical).
        if with_cont:
            cont = np.asarray(response_df['ResponseCont'].values, dtype=np.float32) \
                if 'ResponseCont' in response_df.columns else resp.copy()
            resp = np.stack([resp, cont], axis=1)
        else:
            resp = resp[:, None]
        self.device = device
        if preloaded is not None:
            # A2: gather rows from the factory's shared GPU matrix (no host->device copy,
            # no per-fold pandas reindex). Missing samples -> the trailing zero row.
            padded, row_index = preloaded
            miss = padded.shape[0] - 1
            idx = torch.tensor([row_index.get(s, miss) for s in samples],
                               dtype=torch.long, device=padded.device)
            self.rna = padded.index_select(0, idx)
            self.resp = torch.from_numpy(resp).float().to(padded.device)
            return
        # Vectorized sample -> RNA row resolution (once), missing -> 0.
        rna_mat = rna_df.reindex(samples).to_numpy(dtype=np.float32)
        rna_mat = np.nan_to_num(rna_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.rna = torch.from_numpy(rna_mat).float()
        self.resp = torch.from_numpy(resp).float()
        if device is not None:
            self.rna = self.rna.to(device)
            self.resp = self.resp.to(device)

    def __getitem__(self, index):
        return self.rna[index], self.resp[index]

    def __len__(self):
        return self.rna.shape[0]


class DrugSpecificPairedDataset(Dataset):
    """Balanced paired (cell-line, target) batches with a response-match matrix."""

    def __init__(self, num_samples, batch_size, cl_rna, cl_resp, tg_rna, tg_resp, device=None,
                 cl_preloaded=None, tg_preloaded=None):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.batches = {}

        cl_ds = ResponseDataset(cl_resp, cl_rna.loc[np.unique(cl_resp['Sample'])], device,
                                preloaded=cl_preloaded, with_cont=True)
        tg_ds = ResponseDataset(tg_resp, tg_rna.loc[np.unique(tg_resp['Sample'])], device,
                                preloaded=tg_preloaded, with_cont=True)

        cl_w = get_balanced_class_weights(cl_resp)
        tg_w = get_balanced_class_weights(tg_resp)
        cl_sampler = torch.utils.data.sampler.WeightedRandomSampler(cl_w, num_samples)
        tg_sampler = torch.utils.data.sampler.WeightedRandomSampler(tg_w, num_samples)
        # drop_last guard: a trailing batch of size 1 makes model.py's `batch[0].squeeze()`
        # collapse the batch dim -> 1D tensor -> BatchNorm1d "expected 2D/3D got 1D" (and the
        # 0-d label -> "too many indices"). Drop the last batch ONLY when it would be size 1
        # (num_samples % batch_size < 2), which keeps every other (full) batch. Fixes the
        # small-drug fold failures (e.g. HB3599/Docetaxel/Osimertinib) that blocked coverage.
        _drop_last = (num_samples % batch_size) < 2
        cl_loader = DataLoader(cl_ds, batch_size=batch_size, sampler=cl_sampler, drop_last=_drop_last)
        tg_loader = DataLoader(tg_ds, batch_size=batch_size, sampler=tg_sampler, drop_last=_drop_last)

        def match_fn(dx, dy):
            return torch.stack([dx * y for y in dy]).squeeze() - torch.stack(
                [torch.where(dx != y, torch.ones_like(dx), torch.zeros_like(dx)) for y in dy]).squeeze()

        import math as _math
        def geo_fn(rx, ry):
            # Negative-manifold target Gram: target_cos[i,j] = cos(pi * |rho_i - rho_j|), with
            # rho the continuous responsiveness in [0,1]. Geodesic arccos(cos)=pi*|drho| is thus
            # linearly proportional to the delta in AUC/rank responsiveness. [n_cl, n_tg].
            d = torch.abs(rx.reshape(-1, 1) - ry.reshape(1, -1)).clamp(0.0, 1.0)
            return torch.cos(_math.pi * d)

        for bid, ((cs, cr), (ts, tr)) in enumerate(zip(cl_loader, tg_loader)):
            # cr/tr are [B,2]: col0 binary Response, col1 continuous ResponseCont (with_cont=True).
            cr_bin, cr_cont = cr[:, 0:1], cr[:, 1]
            tr_bin, tr_cont = tr[:, 0:1], tr[:, 1]
            m = match_fn(cr_bin, tr_bin)
            m_geo = geo_fn(cr_cont, tr_cont)
            self.batches[bid] = (cs, ts,
                                 m.to(device) if device is not None else m,
                                 cr_bin.to(device) if device is not None else cr_bin,
                                 tr_bin.to(device) if device is not None else tr_bin,
                                 m_geo.to(device) if device is not None else m_geo)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        return self.batches[index]


def get_data_generator(dataloader):
    return None if dataloader is None else dataloader.__iter__()


# Backwards-compat alias: copied compute modules import the old factory name.
ResponseDataloadersFactory = UnifiedDataloaderFactory
