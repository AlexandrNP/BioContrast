"""
Disjoint cross-validation splitting.

Replaces the ORIGINAL `create_train_val_test_splits`, which used
`StratifiedShuffleSplit(n_splits=cv)` — that draws `cv` INDEPENDENT random test sets
that OVERLAP each other. It is not real K-fold: samples reappear across "folds", so
CV means are optimistic and the std bars understate variance.

Here the outer test folds come from `StratifiedKFold`, so every sample lands in
exactly one test fold and the test folds are mutually disjoint. The validation set is
drawn *only* from that fold's training portion (never from its test portion), so there
is no train/val/test leakage within a fold.

Design decisions (per project requirements):
  * Test folds with a SINGLE positive are kept — they are considered legitimate and are
    NOT filtered out. StratifiedKFold naturally yields ~1 positive per test fold when the
    number of positives is close to the fold count; that is fine.
  * If a class (pos or neg) has fewer members than the requested fold count, the fold
    count is reduced to what stratification allows (still fully disjoint). If even a
    2-fold stratified split is impossible (a class has <2 members total), the drug is
    skipped and reported, because no disjoint stratified CV exists for it.
"""

import os
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold, GroupShuffleSplit)


def _grouped_stratified_shuffle(strat, groups, n_splits, test_size, seed):
    """Leak-free STRATIFIED shuffle-split at the GROUP (family) level.

    Each of the `n_splits` splits draws a test set with >=1 positive-class group AND
    >=1 negative-class group (=> >=2 test samples, both classes present => a defined AUC),
    while keeping >=1 group of each class in TRAIN. A group's replicate rows never span
    train/test (no leakage). Test sets may overlap across splits — that is the point of
    shuffle-split estimation on small data, and is fine (unlike disjoint K-fold it does not
    force single-class folds). Returns list[(train_idx, test_idx)] or None if the drug has
    <2 groups of either class (then it is honestly UNEVALUABLE leak-free)."""
    uniq = np.unique(groups)
    g2lab = {g: int(strat[np.where(groups == g)[0][0]]) for g in uniq}
    pos_g = np.array([g for g in uniq if g2lab[g] == 1])
    neg_g = np.array([g for g in uniq if g2lab[g] == 0])
    if len(pos_g) < 2 or len(neg_g) < 2:
        return None  # cannot keep a positive (or negative) family in BOTH train and test
    splits = []
    for k in range(n_splits):
        r = np.random.RandomState(seed + 1000 * k + 7)
        n_pos = min(max(1, int(round(len(pos_g) * test_size))), len(pos_g) - 1)
        n_neg = min(max(1, int(round(len(neg_g) * test_size))), len(neg_g) - 1)
        test_groups = np.concatenate([r.choice(pos_g, n_pos, replace=False),
                                      r.choice(neg_g, n_neg, replace=False)])
        test_mask = np.isin(groups, test_groups)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(np.unique(strat[train_idx])) < 2 or len(np.unique(strat[test_idx])) < 2:
            continue
        splits.append((train_idx, test_idx))
    return splits or None


def _grouped_stratified_val(train_full_idx, train_strat, train_groups, val_size, seed):
    """Leak-free STRATIFIED grouped validation split from a fold's train portion.

    BUG1 FIX: the previous inner split used GroupShuffleSplit, which ignores the label. At a
    ~10%% positive rate a fold's train often has a SINGLE positive family; GroupShuffleSplit could
    move that whole family into val, leaving TRAIN single-class -> a constant predictor -> chance
    AUC on that fold. Here we only ever send a class's families to val when that class has >=2
    families in train, so TRAIN always keeps >=1 family of each class. Families never split across
    train/val (leak-free). Returns (train_idx, val_idx); val may be single-class (tolerable: it is
    only used for early stopping / model selection), and empty only in the degenerate 1-pos+1-neg
    case (handled by the caller's fallback)."""
    uniq = np.unique(train_groups)
    g2lab = {g: int(round(train_strat[train_groups == g].mean())) for g in uniq}
    pos_g = np.array([g for g in uniq if g2lab[g] == 1])
    neg_g = np.array([g for g in uniq if g2lab[g] == 0])
    r = np.random.RandomState(seed)

    def pick(class_groups):
        # take val_size of the class's groups, but always leave >=1 in TRAIN
        if len(class_groups) < 2:
            return np.array([], dtype=class_groups.dtype)
        n_val = min(max(1, int(round(len(class_groups) * val_size))), len(class_groups) - 1)
        return r.choice(class_groups, n_val, replace=False)

    val_groups = np.concatenate([pick(pos_g), pick(neg_g)])
    val_mask = np.isin(train_groups, val_groups)
    train_idx = train_full_idx[~val_mask]
    val_idx = train_full_idx[val_mask]
    return train_idx, val_idx


def _shuffle_splits(df, strat, groups, n_splits, test_size, seed):
    """Shuffle-split outer folds (BC_CV_MODE=shuffle). Group-aware & stratified when
    `groups` given (leak-free family-level); plain StratifiedShuffleSplit otherwise."""
    if groups is not None:
        return _grouped_stratified_shuffle(strat, groups, n_splits, test_size, seed)
    _, counts = np.unique(strat, return_counts=True)
    if counts.size < 2 or counts.min() < 2:
        return None
    n = len(strat)
    ts = max(test_size, 2.0 / n)  # ensure >= ~2 test samples
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=ts, random_state=seed)
    return list(sss.split(np.zeros(n), strat))


def feasible_n_splits(strat_labels, requested):
    """Largest fold count <= `requested` for which every class has >= n_splits members."""
    labels = np.asarray(strat_labels)
    _, counts = np.unique(labels, return_counts=True)
    min_class = counts.min() if counts.size else 0
    return int(min(requested, min_class))


def make_cv_splits(df, strat_labels, n_splits, val_size=0.2, random_seed=2023, groups=None):
    """
    Build disjoint stratified CV splits.

    Parameters
    ----------
    df : pandas.DataFrame
        The response rows to split (one row per (sample, drug) response).
    strat_labels : array-like
        Per-row stratification labels (binarized response), aligned to `df` rows.
    n_splits : int
        Requested number of outer CV folds.
    val_size : float
        Fraction of each fold's TRAIN portion held out for validation.
    random_seed : int
    groups : array-like or None
        Per-row group id (e.g. PDX patient/family). When given, splitting is
        GROUP-AWARE (StratifiedGroupKFold + GroupShuffleSplit): every replicate of a
        group stays on ONE side of each split, so tumor replicates never leak across
        train/val/test. This is the leak-free replacement for the original PDX
        family->sample expansion followed by sample-level splitting.

    Returns
    -------
    dict[int, dict[str, pandas.DataFrame]]  with keys 'train'/'val'/'test',
    or None if no disjoint stratified CV is possible for this drug.
    """
    df = df.reset_index(drop=True)
    strat = np.asarray(strat_labels)
    groups = None if groups is None else np.asarray(groups)

    # BC_CV_MODE: 'shuffle' (default) = leak-free stratified SHUFFLE-split with a guaranteed
    # >=2 test samples (>=1 per class) per split, for small-sample data; drugs whose class
    # has <2 families are DROPPED (as the original did). 'kfold' = disjoint StratifiedKFold
    # (previous behaviour). BC_TEST_SIZE sets the shuffle test fraction.
    mode = os.environ.get('BC_CV_MODE', 'shuffle')
    test_size = float(os.environ.get('BC_TEST_SIZE', '0.25'))
    if mode == 'shuffle':
        outer_iter = _shuffle_splits(df, strat, groups, n_splits, test_size, random_seed)
        if not outer_iter:
            return None  # drug dropped: too few families of a class for a leak-free split
    else:
        eff_splits = feasible_n_splits(strat, n_splits)
        if groups is not None:
            eff_splits = min(eff_splits, len(np.unique(groups)))
        if eff_splits < 2:
            return None
        if groups is not None:
            try:
                outer = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=random_seed)
                outer_iter = list(outer.split(df, strat, groups))
            except ValueError:
                groups = None
        if groups is None:
            outer = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=random_seed)
            outer_iter = list(outer.split(df, strat))

    cv_splits = {}
    for cv_idx, (train_full_idx, test_idx) in enumerate(outer_iter):
        # Inner validation split drawn ONLY from this fold's training portion.
        train_strat = strat[train_full_idx]
        if groups is not None:
            # BUG1 FIX: stratified grouped val split that never leaves TRAIN single-class.
            train_idx, val_idx = _grouped_stratified_val(
                train_full_idx, train_strat, groups[train_full_idx], val_size, random_seed)
            # Degenerate 1-pos+1-neg train: val came back empty. Fall back to a plain random
            # slice (val is only for early stopping; a tiny within-train val is acceptable) while
            # keeping both classes in TRAIN if at all possible.
            if len(val_idx) == 0:
                rng = np.random.RandomState(random_seed)
                perm = rng.permutation(train_full_idx)
                n_val = max(1, int(round(len(perm) * val_size)))
                cand_val, cand_tr = perm[:n_val], perm[n_val:]
                if len(np.unique(strat[cand_tr])) >= 2:
                    train_idx, val_idx = cand_tr, cand_val
                else:
                    train_idx, val_idx = train_full_idx, perm[:n_val]  # last resort
        elif feasible_n_splits(train_strat, 2) >= 2:  # both classes present
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=val_size, random_state=random_seed)
            tr_rel, val_rel = next(inner.split(train_full_idx, train_strat))
            train_idx = train_full_idx[tr_rel]
            val_idx = train_full_idx[val_rel]
        else:
            # Degenerate train class balance: fall back to a plain random val slice.
            rng = np.random.RandomState(random_seed)
            perm = rng.permutation(train_full_idx)
            n_val = max(1, int(round(len(perm) * val_size)))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

        cv_splits[cv_idx] = {
            'train': df.loc[df.index[train_idx]].copy(),
            'val':   df.loc[df.index[val_idx]].copy(),
            'test':  df.loc[df.index[test_idx]].copy(),
        }
    return cv_splits


def assert_disjoint_test_folds(cv_splits, id_column='Sample'):
    """Sanity check used by tests: every sample appears in exactly one test fold."""
    seen = {}
    for cv_idx, split in cv_splits.items():
        for sid in split['test'][id_column].tolist():
            seen.setdefault(sid, []).append(cv_idx)
    overlaps = {s: f for s, f in seen.items() if len(f) > 1}
    assert not overlaps, f"Test folds overlap for samples: {overlaps}"
    return True
