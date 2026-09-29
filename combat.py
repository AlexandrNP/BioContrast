"""
ComBat batch-effect correction (parametric empirical Bayes; Johnson, Li & Rabinovic 2007).

Used for PDO expression, where samples come from several datasets of origin (studies).
Those studies are strong batch effects; ComBat removes the batch component while keeping
the biological signal. It is UNSUPERVISED (never sees the drug-response label), so it is
applied to the whole expression matrix and does not leak the outcome.

Self-contained NumPy implementation (no `sva`/`pycombat` dependency), samples-as-rows API.

Notes
-----
* Genes with zero variance within any batch are protected (delta floored) to avoid /0.
* Singleton batches (1 sample) cannot yield a variance estimate; their delta is set to 1
  (location-only correction for those batches).
* `ref_batch` optionally anchors the correction to one dataset of origin (no shift for it).
"""

import numpy as np
import pandas as pd


def _aprior(delta_hat):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_it=200):
    """Iterative EB solution for one batch (genes x samples_in_batch)."""
    n = (~np.isnan(s_data)).sum(axis=1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    change = 1.0
    it = 0
    while change > conv and it < max_it:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        resid = s_data - g_new[:, None]
        sum2 = np.nansum(resid ** 2, axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                     np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
        g_old, d_old = g_new, d_new
        it += 1
    return g_old, d_old


def combat(expr, batch, ref_batch=None, parametric=True):
    """
    Parameters
    ----------
    expr : pandas.DataFrame  (samples x genes)   -- rows indexed by sample id
    batch : array-like of length n_samples       -- dataset of origin per sample
    ref_batch : optional hashable                 -- batch left unshifted
    parametric : bool                             -- parametric EB (True) or simple (False)

    Returns
    -------
    pandas.DataFrame with the same index/columns, batch-corrected.
    """
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("combat expects a samples x genes DataFrame")

    genes = expr.columns
    index = expr.index
    X = expr.to_numpy(dtype=np.float64).T          # genes x samples
    batch = np.asarray(batch)
    if batch.shape[0] != X.shape[1]:
        raise ValueError("batch length must equal number of samples (rows of expr)")

    batches = pd.unique(batch)
    batch_idx = {b: np.where(batch == b)[0] for b in batches}
    n_array = X.shape[1]

    # Drop genes with no variance across the whole matrix (nothing to correct, avoids /0).
    gvar = np.nanvar(X, axis=1)
    keep = gvar > 1e-12
    Xk = X[keep]

    # --- standardise across samples using a batch-size-weighted grand mean/var ---
    design = np.zeros((n_array, len(batches)))
    for j, b in enumerate(batches):
        design[batch_idx[b], j] = 1.0
    sizes = design.sum(axis=0)

    B = np.linalg.lstsq(design, Xk.T, rcond=None)[0]     # (n_batches x genes) batch means
    grand_mean = (sizes / n_array) @ B                   # weighted grand mean per gene
    resid = Xk - (design @ B).T
    var_pooled = (resid ** 2).mean(axis=1)
    var_pooled = np.where(var_pooled < 1e-12, 1e-12, var_pooled)
    sd = np.sqrt(var_pooled)

    stand = (Xk - grand_mean[:, None]) / sd[:, None]

    # --- estimate batch location (gamma) and scale (delta) with EB shrinkage ---
    gamma_star = np.zeros((len(batches), Xk.shape[0]))
    delta_star = np.ones((len(batches), Xk.shape[0]))

    for j, b in enumerate(batches):
        idx = batch_idx[b]
        s_data = stand[:, idx]
        n_b = idx.size
        g_hat = np.nanmean(s_data, axis=1)
        if n_b < 2:
            gamma_star[j] = g_hat
            delta_star[j] = 1.0
            continue
        d_hat = np.nanvar(s_data, axis=1, ddof=1)
        d_hat = np.where(d_hat < 1e-12, 1e-12, d_hat)
        g_bar, t2 = g_hat.mean(), g_hat.var()
        a, bp = _aprior(d_hat), _bprior(d_hat)
        if parametric:
            g_star, d_star = _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, bp)
        else:
            g_star, d_star = g_hat, d_hat
        gamma_star[j] = g_star
        delta_star[j] = np.where(d_star < 1e-12, 1e-12, d_star)

    if ref_batch is not None and ref_batch in list(batches):
        rj = list(batches).index(ref_batch)
        gamma_star[rj] = 0.0
        delta_star[rj] = 1.0

    # --- adjust ---
    adjusted = stand.copy()
    for j, b in enumerate(batches):
        idx = batch_idx[b]
        adjusted[:, idx] = (stand[:, idx] - gamma_star[j][:, None]) / np.sqrt(delta_star[j])[:, None]
    adjusted = adjusted * sd[:, None] + grand_mean[:, None]

    out = X.copy()
    out[keep] = adjusted
    # Guard against any non-finite values leaking downstream (they would turn the
    # contrastive loss into NaN). Replace inf/nan with 0.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(out.T, index=index, columns=genes)
