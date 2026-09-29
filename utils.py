import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
import numpy as np
import pandas as pd

def l1_norm(model):
    l1 = 0
    for param in model.parameters():
        l1 += torch.abs(param.view(-1)).sum()
    return l1


def orthogonality_penalty(Z, C):
    """Squared Frobenius norm of the linear cross-covariance between the embedding Z and
    the confound score vectors C, normalized by n^2.

    Z : [n, d] embedding batch.  C : [n, k] confound scores for the SAME n samples.
    Both are column-centered; the penalty is ||Zc^T Cc||_F^2 / n^2, which is 0 iff every
    embedding dimension is (empirically) linearly uncorrelated with every confound score.
    Driving this toward 0 during training removes any linear signal about the confounds
    (proliferation / EMT / IFN) from the contrastive embedding.
    """
    if Z is None or C is None:
        return None
    n = Z.shape[0]
    if n < 2:
        return Z.new_zeros(())
    C = C.to(dtype=Z.dtype, device=Z.device)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    Cc = C - C.mean(dim=0, keepdim=True)
    cross_cov = Zc.transpose(0, 1) @ Cc            # [d, k]
    return (cross_cov ** 2).sum() / (n * n)


def confound_scores_from_batch(X, cols, gene_mu, gene_sd, mod_mu, mod_sd, clip=6.0):
    """Compute per-sample confound scores ON THE FLY from a batch's raw expression X.

    Mirrors build_confounds.py exactly (robust per-gene z-score, mean of member genes,
    then standardize each module score) but in torch so it stays on-graph / on-device.

    X       : [n, G] raw expression batch (gene columns in genes.npy order).
    cols    : list of LongTensor gene-index vectors, one per confound module.
    gene_mu, gene_sd : [G] robust per-gene mean/std (this modality's non-outlier samples).
    mod_mu, mod_sd   : [k] per-module score mean/std used to standardize each column.
    Returns C : [n, k] (detached from the input's grad; scores are treated as fixed targets).
    """
    X = X.detach()
    Xz = torch.clamp((X - gene_mu) / gene_sd, -clip, clip)
    scores = []
    for j, idx in enumerate(cols):
        if idx.numel() == 0:
            scores.append(torch.zeros(X.shape[0], device=X.device, dtype=X.dtype))
        else:
            s = Xz.index_select(1, idx).mean(dim=1)
            scores.append((s - mod_mu[j]) / mod_sd[j])
    return torch.stack(scores, dim=1)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def nt_xent_loss(embedding1, embedding2, temperature):
    # Use gram matrix to calculate loss
    logits = (embedding1 @ embedding2.T) / temperature
    embedding1_similarity = embedding1 @ embedding1.T
    embedding2_similarity = embedding2 @ embedding2.T
    targets = F.softmax(
        (embedding1_similarity + embedding2_similarity) / (2*temperature), dim=-1)

    cell_line_loss = cross_entropy(logits, targets, reduction='none')
    pdx_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (cell_line_loss + pdx_loss) / 2.

    return loss.mean()


def nt_bnext_loss(embeddings1, embeddings2, labels, temperature, alpha, device='cpu'):
    eps = 1e-6
    # logits = (embeddings1 @ embeddings2.T) / temperature
    assert embeddings1.size(0) == embeddings2.size(0)
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))

    # logits = embeddings1[cartesian_indices[:,0]] * embeddings2[cartesian_indices[:,1]] /  \
    #    (norm(embeddings1[[cartesian_indices[:,0]]], axis=0) * norm(embeddings2[[cartesian_indices[:,1]]], axis=0) )
    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = norm1[cartesian_indices[:, 0]] * \
        norm2[cartesian_indices[:, 1]]
    cosine_norm = torch.reshape(cosine_norm, (num_samples, num_samples))
    logits = logits / (cosine_norm*temperature)

    # logits = #(embeddings1 @ embeddings2.T) / (vector_norm(embeddings1, ord=2) * vector_norm(embeddings2, ord=2) * temperature)
    # cosine_sim = F.cosine_similarity(embeddings1, embeddings2.T)
    # logits = cosine_sim / temperature

    # loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    match = labels.bool().to(device)

    def first_non_zero(matrix):
        # Ensure the matrix is a PyTorch tensor
        eps = 1e-6

        # Create a mask of non-zero elements
        non_zero_mask = torch.abs(matrix) > eps

        # Find indices of the first non-zero element in each row
        first_non_zero_indices = torch.argmax(non_zero_mask.long(), dim=1)

        # Extract the first non-zero element from each row
        first_non_zero_elements = matrix[torch.arange(
            matrix.size(0)), first_non_zero_indices]

        return first_non_zero_elements

    anchors = first_non_zero(logits*match)

    mismatch = ~labels.bool().to(device)

    # loss_match = torch.zeros(num_samples, num_samples).to(
    #    device).masked_scatter(match, loss[match]).to(device)

    # loss_mismatch = torch.zeros(num_samples, num_samples).to(
    #    device).masked_scatter(mismatch, loss[mismatch]).to(device)

    # loss_match = loss_match.sum(dim=-1)
    # loss_mismatch = loss_mismatch.sum(dim=-1)
    num_matches = torch.sum(match)
    num_matches_per_row = torch.sum(match, axis=0)
    nonzero_rows = torch.where(num_matches_per_row > eps)[0]
    # num_mismatches = torch.sum(mismatch)

    alignment = torch.negative(
        torch.sum(torch.sum(logits*match, axis=1)/(num_matches_per_row+1)))

    logits_with_anchors = torch.cat((logits, anchors.unsqueeze(1)), dim=1)
    mismatch_with_anchors = torch.cat(
        (mismatch, torch.ones(anchors.shape[0]).unsqueeze(1).to(device)), dim=1)

    log_exp_logits = torch.logsumexp(
        logits_with_anchors*mismatch_with_anchors, axis=1)
    uniformity = torch.sum(
        log_exp_logits / (num_matches_per_row+1))
    # uniformity = sum(log_exp_logits)
    # breakpoint()

    rowwise_attempt = """
    alignment = torch.negative(
        torch.sum(torch.sum(logits[match], axis=0)/num_matches_per_row))
    # uniformity = torch.logsumexp(logits, (0, 1))/num_matches

    exp_logits = torch.exp(logits)
    rowise_negatives = torch.sum(exp_logits*mismatch, axis=0)

    # How to add row-wise sum to the logit matrix
    exp_logits_with_negative_sum = (exp_logits[:, :,
                                               None]+rowise_negatives[:, None, None]).squeeze()

    # logits[match] + logits[match]*

    uniformity = torch.sum(torch.sum(
        torch.log(exp_logits_with_negative_sum[match]), axis=0) / num_matches_per_row)

    # uniformity = torch.sum(torch.logsumexp(
    #    logits*torch.log(1/num_matches), 0))

    # (alpha * (loss_match/num_matches) + (1-alpha)*(loss_match/num_mismatches)).mean()
    """
    return alignment + uniformity

    # embedding1_similarity = embedding1 @ embedding1.T
    # embedding2_similarity = embedding2 @ embedding2.T


def sup_con_mod(embeddings1, embeddings2, labels, temperature, alpa=0.5, device='cpu'):
    eps = 1e-6
    assert embeddings1.size(0) == embeddings2.size(0)
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))

    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = norm1[cartesian_indices[:, 0]] * \
        norm2[cartesian_indices[:, 1]]
    cosine_norm = torch.reshape(cosine_norm, (num_samples, num_samples))
    logits = logits / (cosine_norm*temperature)

    match = labels.bool().to(device)
    mismatch = ~labels.bool().to(device)

    alignment = torch.negative(torch.sum(logits*match)/torch.sum(match))
    uniformity = torch.sum(logits*mismatch)/torch.sum(mismatch)
    # breakpoint()
    return alignment + uniformity


def sup_con(embeddings, labels, temperature, device='cpu'):
    # labels is a 1-D vector with corresponding class labels
    eps = 1e-6
    num_samples = embeddings.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))
    logits = embeddings @ embeddings.T
    norm1 = norm(embeddings, axis=1, ord=2)


def sup_con_cross_modal_rowise(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    eps = 1e-6
    assert embeddings1.size(0) == embeddings2.size(0)
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))

    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = norm1[cartesian_indices[:, 0]] * \
        norm2[cartesian_indices[:, 1]]
    cosine_norm = torch.reshape(cosine_norm, (num_samples, num_samples))
    logits = logits / (cosine_norm*temperature)

    classes = torch.unique(labels)
    class_dependent_match = [torch.where(labels == y, torch.ones_like(
        labels), torch.zeros_like(labels)) for y in classes]
    # breakpoint()

    def supervised_loss_in_row(i, class_labels, logits):
        class_labels = class_labels > 0
        num_positives = torch.sum(class_labels[i])
        if num_positives < 1 or num_positives == num_samples:
            return 0
        alignment = torch.negative(
            torch.sum(logits[i]*class_labels[i])/num_positives)
        mismatch = ~class_labels
        mismatch_logits = logits*mismatch
        uniformity = 0
        for j in range(class_labels.size()[1]):
            if class_labels[i][j]:
                to_sum = torch.cat(
                    (logits[i][j].unsqueeze(dim=0), mismatch_logits[i]))
                uniformity += torch.logsumexp(
                    to_sum[to_sum.nonzero(as_tuple=True)], dim=0)
        uniformity /= num_positives
        return alignment + uniformity

    loss = 0
    for class_labels in class_dependent_match:
        # match = class_labels.bool().to(device)
        # mismatch = ~class_labels.bool().to(device)

        n, m = class_labels.size()
        for i in range(n):
            loss += supervised_loss_in_row(i, class_labels, logits)

    return loss


# MODIFYING CONTRASTIVE LOSS FUNCTION TO BE PERFORMANT
def sup_con_cross_modal_rowise_fast(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    assert embeddings1.size(0) == embeddings2.size(0)
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))

    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = norm1[cartesian_indices[:, 0]] * \
        norm2[cartesian_indices[:, 1]]
    cosine_norm = torch.reshape(cosine_norm, (num_samples, num_samples))
    logits = logits / (cosine_norm*temperature)

    classes = torch.unique(labels)
    class_dependent_match = [torch.where(
        labels == y, True, False) for y in classes]
    # breakpoint()

    def compute_loss(mask, inv_mask, logits, axis=-1):
        # print(mask)
        # print(logits)
        # print([
        #    torch.logsumexp(x_i*mask_i, -1)/sum(mask_i)-torch.logsumexp(x_i*inv_mask_i, -1) if sum(mask_i) > 0 and sum(mask_i) < num_samples else torch.zeros(1).to(device)[0] for x_i, mask_i, inv_mask_i in zip(torch.unbind(logits, dim=axis), torch.unbind(mask, dim=axis), torch.unbind(inv_mask, dim=axis))
        # ])
        # print(torch.stack([
        #    torch.logsumexp(x_i*mask_i, -1)/sum(mask_i)-torch.logsumexp(x_i*inv_mask_i, -1) if sum(mask_i) > 0 and sum(mask_i) < num_samples else torch.zeros(1).to(device)[0] for x_i, mask_i, inv_mask_i in zip(torch.unbind(logits, dim=axis), torch.unbind(mask, dim=axis), torch.unbind(inv_mask, dim=axis))
        # ], dim=axis))
        #
        #  breakpoint()
        # Proper SupCon masking: exclude non-members with a large-negative fill so they drop
        # out of the log-sum-exp. The original `x_i*mask_i` set non-members to logit 0 ->
        # exp(0)=1, leaking a constant similarity into both numerator and denominator and
        # blunting the positive/negative separation the encoder is supposed to learn.
        import os as _os
        _mask_mode = _os.environ.get('BC_LOSS_MASK', 'fixed')   # fixed | old
        NEG = -1e9
        rows = []
        for x_i, mask_i, inv_mask_i in zip(torch.unbind(logits, dim=axis),
                                           torch.unbind(mask, dim=axis),
                                           torch.unbind(inv_mask, dim=axis)):
            cnt = mask_i.sum()
            if cnt > 0 and cnt < num_samples:
                if _mask_mode == 'old':
                    rows.append(torch.negative(torch.log(1. / cnt)
                                               + torch.logsumexp(x_i * mask_i, -1)
                                               - torch.logsumexp(x_i * inv_mask_i, -1)))
                else:
                    pos = torch.where(mask_i, x_i, torch.full_like(x_i, NEG))
                    neg = torch.where(inv_mask_i, x_i, torch.full_like(x_i, NEG))
                    rows.append(torch.negative(-torch.log(cnt.float())
                                               + torch.logsumexp(pos, -1)
                                               - torch.logsumexp(neg, -1)))
            else:
                rows.append(torch.zeros((), device=device))
        return torch.mean(torch.stack(rows))
        # return torch.mean(torch.stack([
        #    torch.negative(torch.logsumexp(x_i*mask_i*(float(sum(inv_mask_i))/sum(mask_i)), -1)-torch.logsumexp(x_i*inv_mask_i*(float(sum(inv_mask_i))/sum(mask_i)), -1)) if sum(mask_i) > 0 and sum(mask_i) < num_samples else torch.zeros(1).to(device)[0] for x_i, mask_i, inv_mask_i in zip(torch.unbind(logits, dim=axis), torch.unbind(mask, dim=axis), torch.unbind(inv_mask, dim=axis))
        # ], dim=axis))

    loss = 0
    # breakpoint()
    for class_labels in class_dependent_match:
        loss += compute_loss(class_labels, ~class_labels, logits)
        # print(loss)

    return loss


# Nevermind, it is equivalent to applying sup_con_cross_modal_rowise twice - to the original and transposed matrices
def sup_con_cross_modal_all_data(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    eps = 1e-6
    assert embeddings1.size(0) == embeddings2.size(0)
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(
        start=0, end=num_samples), torch.arange(start=0, end=num_samples))

    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = norm1[cartesian_indices[:, 0]] * \
        norm2[cartesian_indices[:, 1]]
    cosine_norm = torch.reshape(cosine_norm, (num_samples, num_samples))
    logits = logits / (cosine_norm*temperature)

    class_dependent_match = [labels.bool(), ~labels.bool()]

    def supervised_loss(i, j, class_labels, logits):
        class_labels_ij = class_labels.clone()
        class_labels_ij[i][j] = 0
        num_positives = torch.sum(
            class_labels[i][:]) + torch.sum(class_labels[:][j])
        if num_positives < 1 or num_positives == num_samples:
            return 0
        alignment = torch.negative(torch.sum(torch.cat(
            logits[i][:]*class_labels[i][:], logits[:][j]*class_labels[:][j], dim=0))/num_positives)

        # NOT using class_labels_ij to avoid summing element i
        # mismatch[i][j] should be 0
        mismatch = ~class_labels
        mismatch_logits = logits*mismatch
        uniformity = 0
        for k in range(class_labels.size()[1]):
            if class_labels[i][k]:
                to_sum = torch.cat((logits[i][j].unsqueeze(
                    dim=0), mismatch_logits[i][:], mismatch_logits[:][j]))
                uniformity += torch.logsumexp(
                    to_sum[to_sum.nonzero(as_tuple=True)], dim=0)

        for k in range(class_labels.size()[0]):
            pass

        uniformity /= num_positives
        return alignment + uniformity

    loss = 0
    for class_labels in class_dependent_match:
        # match = class_labels.bool().to(device)
        # mismatch = ~class_labels.bool().to(device)

        n, m = class_labels.size()
        non_zero_labels = class_labels.nonzero()
        # for i, j in ...
        #    loss += supervised_loss_in_row(i, class_labels, logits)

    return loss


def anchorless_sup_con_loss(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    return nt_bnext_loss(embeddings1, embeddings2, labels, temperature, alpha, device=device) + nt_bnext_loss(embeddings2, embeddings1, labels, temperature, alpha, device=device)


def sup_con_transfer_learning(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    # Contrastive function over two semantically different modalities                                                                                                                                                                                                       fferent modalities is not commutative
    # We need to consider two matching representations - original and transposed - to account for distance between both modalities
    return sup_con_cross_modal_rowise(embeddings1, embeddings2, labels, temperature, alpha=alpha, device=device) +\
        sup_con_cross_modal_rowise(
            embeddings2, embeddings1, labels.T, temperature, alpha=alpha, device=device)


def sup_con_transfer_learning_fast(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    # Contrastive function over two semantically different modalities                                                                                                                                                                                                       fferent modalities is not commutative
    # We need to consider two matching representations - original and transposed - to account for distance between both modalities
    return sup_con_cross_modal_rowise_fast(embeddings1, embeddings2, labels, temperature, alpha=alpha, device=device) +\
        sup_con_cross_modal_rowise_fast(
            embeddings2, embeddings1, labels.T, temperature, alpha=alpha, device=device)


def info_nce_cross_modal_single_pos(embeddings1, embeddings2, labels, temperature, device='cpu'):
    """Plain single-positive InfoNCE (the contrast to multi-positive BioContrast/SupCon).

    For each anchor exactly ONE positive is used: a randomly sampled same-response
    partner (match-matrix label >= 0, i.e. both-responder=+1 or both-non-responder=0);
    the negatives are the mismatched partners (label == -1). Uses the identical cosine
    normalisation + temperature scaling as sup_con_cross_modal_rowise_fast, so the
    loss-type ablation isolates single-positive-InfoNCE vs multi-positive-SupCon."""
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = (norm1[cartesian_indices[:, 0]] * norm2[cartesian_indices[:, 1]]).reshape(
        num_samples, num_samples)
    logits = logits / (cosine_norm * temperature)

    pos_mask = labels >= 0          # same binary response
    neg_mask = labels < 0           # mismatched response
    rows = []
    for i in range(num_samples):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue  # no positive or no negative for this anchor -> skip (undefined InfoNCE)
        p = pos_idx[torch.randint(pos_idx.numel(), (1,), device=logits.device)]
        pos_logit = logits[i, p]                              # (1,) single positive
        cand = torch.cat([pos_logit, logits[i, neg_idx]])     # positive + all negatives
        rows.append(torch.logsumexp(cand, 0) - pos_logit.squeeze())
    if not rows:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.mean(torch.stack(rows))


def info_nce_transfer_learning(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """Symmetrised single-positive InfoNCE across the two modalities (mirror of
    sup_con_transfer_learning_fast, for the BC_LOSS_TYPE=infonce ablation)."""
    # BUG2 FIX: the match matrix arrives as [target][cell_line] (data.py match_fn), but
    # cross_modal(e1,e2,L) requires L[i,j]=rel(e1_i,e2_j). With e1=cell_line, e2=target the
    # first term needs labels.T and the second needs labels (previously swapped -> the row-based
    # losses were fed the transposed orientation and pushed apart pairs they should pull together).
    return info_nce_cross_modal_single_pos(embeddings1, embeddings2, labels.T, temperature, device=device) +\
        info_nce_cross_modal_single_pos(embeddings2, embeddings1, labels, temperature, device=device)


def supcon_std_cross_modal(embeddings1, embeddings2, labels, temperature, device='cpu'):
    """STANDARD supervised contrastive loss (Khosla et al.) for the loss ablation.

    Positives for each anchor = ALL same-response partners (match label >= 0, i.e.
    both-responder or both-non-responder); negatives = mismatches (label == -1). Unlike
    BioContrast (sup_con_cross_modal_rowise_fast) it does NOT treat the mismatch class as
    its own group to cluster -- it only pulls same-response pairs together and pushes
    mismatches apart (the canonical multi-positive SupCon). Same cosine/temperature scaling."""
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = (norm1[cartesian_indices[:, 0]] * norm2[cartesian_indices[:, 1]]).reshape(
        num_samples, num_samples)
    logits = logits / (cosine_norm * temperature)
    mask = labels >= 0           # same-response = positive
    inv_mask = labels < 0        # mismatch = negative
    NEG = -1e9
    rows = []
    for x_i, m_i, im_i in zip(torch.unbind(logits, 0), torch.unbind(mask, 0), torch.unbind(inv_mask, 0)):
        cnt = m_i.sum()
        if cnt > 0 and cnt < num_samples:
            pos = torch.where(m_i, x_i, torch.full_like(x_i, NEG))
            neg = torch.where(im_i, x_i, torch.full_like(x_i, NEG))
            rows.append(torch.negative(-torch.log(cnt.float())
                                       + torch.logsumexp(pos, -1) - torch.logsumexp(neg, -1)))
        else:
            rows.append(torch.zeros((), device=device))
    return torch.mean(torch.stack(rows))


def supcon_std_transfer(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """Symmetrised standard-SupCon across the two modalities (BC_LOSS_TYPE=supcon_std)."""
    # BUG2 FIX (see info_nce_transfer_learning): correct [target][cell_line] -> per-anchor orientation.
    return supcon_std_cross_modal(embeddings1, embeddings2, labels.T, temperature, device=device) +\
        supcon_std_cross_modal(embeddings2, embeddings1, labels, temperature, device=device)


def sup_con_3class_fixed_cross_modal(embeddings1, embeddings2, labels, temperature, device='cpu'):
    """FIXED 3-class biocontrast loss (BC_LOSS_TYPE=supcon3fix).

    The original 3-class loss (sup_con_cross_modal_rowise_fast) is DEGENERATE: it iterates over ALL
    THREE relationship classes {+1 both-responder, 0 both-non-responder, -1 mismatch} and tries to
    CLUSTER each -- including aligning the mismatch pairs. Summing the SupCon term over the three
    COMPLEMENTARY classes cancels the embedding-dependence exactly (aligning mismatched pairs undoes
    the alignment the same-response classes create) -> the loss is constant w.r.t. embeddings and
    temperature -> ~zero contrastive gradient (verified: perfect-cluster == random == const).

    THE FIX: only the two SAME-RESPONSE classes (+1 and 0) are positive groups to cluster (separately,
    so responders and non-responders form two distinct clusters -- the genuine 3-way geometry); the
    mismatch class (-1) is used ONLY as negatives, never aligned. Proper -1e9 masking. This restores a
    real contrastive gradient (perfect-cluster << random; temperature-sensitive)."""
    num_samples = embeddings1.size(0)
    ci = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    logits = embeddings1 @ embeddings2.T
    n1 = norm(embeddings1, axis=1, ord=2); n2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = (n1[ci[:, 0]] * n2[ci[:, 1]]).reshape(num_samples, num_samples)
    logits = logits / (cosine_norm * temperature)
    NEG = -1e9
    loss = 0
    for y in (1.0, 0.0):                       # same-response classes ONLY (mismatch -1 excluded as a positive)
        mask = (labels == y); inv = ~mask
        rows = []
        for x_i, m_i, im_i in zip(torch.unbind(logits, -1), torch.unbind(mask, -1), torch.unbind(inv, -1)):
            cnt = m_i.sum()
            if cnt > 0 and cnt < num_samples:
                pos = torch.where(m_i, x_i, torch.full_like(x_i, NEG))
                neg = torch.where(im_i, x_i, torch.full_like(x_i, NEG))
                rows.append(torch.negative(-torch.log(cnt.float())
                                           + torch.logsumexp(pos, -1) - torch.logsumexp(neg, -1)))
            else:
                rows.append(torch.zeros((), device=device))
        loss = loss + torch.mean(torch.stack(rows))
    return loss


def sup_con_3class_fixed_transfer(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """Symmetrised FIXED 3-class loss across the two modalities (BC_LOSS_TYPE=supcon3fix)."""
    return sup_con_3class_fixed_cross_modal(embeddings1, embeddings2, labels.T, temperature, device=device) + \
        sup_con_3class_fixed_cross_modal(embeddings2, embeddings1, labels, temperature, device=device)


def supcon_geo_cross_modal(embeddings1, embeddings2, target_cos, temperature, device='cpu'):
    """NEGATIVE-MANIFOLD / continuous-response geometry loss (BC_LOSS_TYPE=geo).

    Instead of a binary match/mismatch mask, the DESIRED cosine between cell-line anchor i
    and target j is a continuous function of their responsiveness gap:
        target_cos[i, j] = cos(pi * |rho_i - rho_j|),  rho in [0, 1] (1 = most responsive).
    So the geodesic distance on the unit hypersphere, arccos(cos) = pi * |rho_i - rho_j|, is
    LINEARLY proportional to the delta in AUC-derived responsiveness (PDO/cell-line) or ranked
    responsiveness (PDX). Same-responsiveness pairs coincide (cos=+1); maximally-different pairs
    are antipodal (cos=-1). Reduces to the SupCon geometry at the binary extremes.

    Loss = MSE between the realized cross-modal cosine and this target Gram matrix. Embeddings
    are L2-normalised so the comparison lives on the sphere. `temperature` is accepted for a
    uniform call signature but the geometric MSE does not use it (target is already a cosine)."""
    n1 = embeddings1.size(0)
    n2 = embeddings2.size(0)
    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    denom = torch.outer(norm1, norm2).clamp_min(1e-8)
    cos = logits / denom                                  # [n1, n2] cross-modal cosine
    tgt = target_cos.reshape(n1, n2).to(cos.dtype)
    return torch.mean((cos - tgt) ** 2)


def supcon_geo_transfer(embeddings1, embeddings2, target_cos, temperature, alpha=0.5, device='cpu'):
    """Continuous negative-manifold geometry across the two modalities (BC_LOSS_TYPE=geo).
    target_cos[i, j] is the desired cosine between cell_line_i and target_j; the MSE Gram-matrix
    objective already covers every cross pair, so no transpose/symmetrisation is required."""
    return supcon_geo_cross_modal(embeddings1, embeddings2, target_cos, temperature, device=device)


def ntxent_transfer(embeddings1, embeddings2, temperature):
    """Symmetrised NT-Xent / CLIP-style self-supervised contrastive (BC_LOSS_TYPE=ntxent).
    Uses the existing soft-target nt_xent_loss; no response labels (unsupervised alignment)."""
    return nt_xent_loss(embeddings1, embeddings2, temperature)


def supcon_fixed_cross_modal(embeddings1, embeddings2, labels, temperature, device='cpu'):
    """CORRECTED supervised contrastive loss — fixes the two BioContrast defects:
      (1) does NOT cluster the mismatch class (positives = same-response only, label>=0);
      (2) uses the CANONICAL SupCon-out denominator = log-sum-exp over ALL partners
          (not negatives-only), and averages the log-prob over the positives:
              L_i = -mean_{p in P(i)} [ s_ip/T - logsumexp_j(s_ij/T) ]
    Cosine-normalised, same temperature as the other losses."""
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    logits = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = (norm1[cartesian_indices[:, 0]] * norm2[cartesian_indices[:, 1]]).reshape(
        num_samples, num_samples)
    logits = logits / (cosine_norm * temperature)

    pos_mask = labels >= 0                       # same-response = positive
    rows = []
    for x_i, m_i in zip(torch.unbind(logits, 0), torch.unbind(pos_mask, 0)):
        cnt = m_i.sum()
        if cnt > 0 and cnt < num_samples:
            denom = torch.logsumexp(x_i, -1)     # over ALL partners (canonical SupCon-out)
            pos_logprob = x_i[m_i] - denom       # log p for each positive
            rows.append(-pos_logprob.mean())     # mean over positives
        else:
            rows.append(torch.zeros((), device=device))
    return torch.mean(torch.stack(rows))


def supcon_fixed_transfer(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """Symmetrised corrected SupCon across the two modalities (BC_LOSS_TYPE=supcon_fixed)."""
    # BUG2 FIX (see info_nce_transfer_learning): correct [target][cell_line] -> per-anchor orientation.
    return supcon_fixed_cross_modal(embeddings1, embeddings2, labels.T, temperature, device=device) +\
        supcon_fixed_cross_modal(embeddings2, embeddings1, labels, temperature, device=device)


def supcon_pos_cross_modal(embeddings1, embeddings2, labels, temperature, margin=0.0, device='cpu'):
    """SINGLE-CLASS (positive-focused) supervised contrastive loss — the project THESIS loss.

    Unlike standard SupCon (supcon_fixed, positives = ANY same-response pair, label>=0), this
    pulls together ONLY responder<->responder pairs (label==+1) and pushes everything else away,
    concentrating the objective on tightening the margin of the scarce POSITIVE (responder) class
    during transfer. Non-responder anchors (no positive partner) contribute no pull term; they
    still serve as negatives for responder anchors via the full-set denominator.

    An optional additive angular `margin` (subtracted from the cosine of positive pairs before
    the temperature scaling, CosFace-style) makes the responder cluster tighter/harder. Denominator
    is the canonical log-sum-exp over ALL partners. `labels` must be per-anchor oriented
    (labels[i,j] describes (e1_i, e2_j)); the transfer wrapper handles orientation."""
    num_samples = embeddings1.size(0)
    cartesian_indices = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    sim = embeddings1 @ embeddings2.T
    norm1 = norm(embeddings1, axis=1, ord=2)
    norm2 = norm(embeddings2, axis=1, ord=2)
    cosine_norm = (norm1[cartesian_indices[:, 0]] * norm2[cartesian_indices[:, 1]]).reshape(
        num_samples, num_samples)
    cos = sim / cosine_norm                       # cosine similarity in [-1, 1]
    pos_mask = labels == 1                         # responder<->responder ONLY (single class)
    # CosFace-style additive margin: penalise positive cosines so they must exceed negatives by `margin`.
    cos_m = cos - margin * pos_mask.float()
    logits = cos_m / temperature
    rows = []
    for x_i, m_i in zip(torch.unbind(logits, 0), torch.unbind(pos_mask, 0)):
        cnt = m_i.sum()
        if cnt > 0 and cnt < num_samples:
            denom = torch.logsumexp(x_i, -1)      # over ALL partners (canonical)
            pos_logprob = x_i[m_i] - denom
            rows.append(-pos_logprob.mean())
        else:
            rows.append(torch.zeros((), device=device))
    return torch.mean(torch.stack(rows))


def supcon_pos_transfer(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """Symmetrised single-class positive-focused SupCon (BC_LOSS_TYPE=supcon_pos).
    BC_POS_MARGIN sets the additive angular margin (default 0.1). BUG2-correct orientation."""
    margin = float(os.environ.get('BC_POS_MARGIN', '0.1'))
    return supcon_pos_cross_modal(embeddings1, embeddings2, labels.T, temperature, margin, device=device) +\
        supcon_pos_cross_modal(embeddings2, embeddings1, labels, temperature, margin, device=device)


def biocontrast_occ_cross_modal(embeddings1, embeddings2, labels, temperature, device='cpu'):
    """PROPER BioContrast (ICML paper, eqs 4-6): the single-class / one-class-classification (OCC)
    supervised contrastive objective. Per the paper: "alignment ONLY between responder samples".

      A (alignment)  = -1/|P| * sum_{x+ in P} sigma(xi, x+)          [attract responder<->responder]
      U (uniformity) =  1/|P| * sum_{x+ in P} log[ exp(sigma(xi,x+)) + sum_{x- in N} exp(sigma(xi,x-)) ]
      L = mean over RESPONDER anchors of (A + U);  non-responder anchors contribute 0 (A=0).

    Differences from supcon_pos (which was the closest existing variant): (1) NO CosFace margin;
    (2) per-positive denominator = that positive + the NEGATIVES only (label==-1 mismatches), NOT the
    other positives (canonical OCC 'SupCon-in' denominator, exactly eq 5). sigma = cosine / temperature.
    Positives = responder<->responder (label==1); negatives = mismatch (label==-1). `labels` per-anchor
    oriented (labels[i,j] describes (e1_i, e2_j)); the transfer wrapper handles orientation."""
    num_samples = embeddings1.size(0)
    cart = torch.cartesian_prod(torch.arange(num_samples), torch.arange(num_samples))
    sim = embeddings1 @ embeddings2.T
    n1 = norm(embeddings1, axis=1, ord=2)
    n2 = norm(embeddings2, axis=1, ord=2)
    cnorm = (n1[cart[:, 0]] * n2[cart[:, 1]]).reshape(num_samples, num_samples)
    cos = sim / cnorm / temperature                       # cosine similarity, temperature-scaled
    pos = (labels == 1)                                    # responder <-> responder (single positive class)
    neg = (labels == -1)                                   # mismatch = negatives
    rows, is_resp = [], []
    for x_i, p_i, ng_i in zip(torch.unbind(cos, 0), torch.unbind(pos, 0), torch.unbind(neg, 0)):
        if p_i.sum() > 0:                                  # RESPONDER anchor only (OCC single-class)
            pos_sims = x_i[p_i]
            neg_sims = x_i[ng_i]
            if neg_sims.numel() > 0:
                neg_lse = torch.logsumexp(neg_sims, 0)                       # log sum_neg exp(sigma)
                denom = torch.logaddexp(pos_sims, neg_lse.expand_as(pos_sims))  # log(exp(pos_j)+sum_neg)
            else:
                denom = pos_sims                            # no negatives present
            rows.append((-pos_sims + denom).mean())         # (A_j + U_j) averaged over positives
            is_resp.append(1.0)
        else:
            rows.append(torch.zeros((), device=device)); is_resp.append(0.0)
    stacked = torch.stack(rows)
    m = torch.tensor(is_resp, device=stacked.device)
    return (stacked * m).sum() / m.sum().clamp(min=1.0)     # mean over responder anchors


def biocontrast_transfer(embeddings1, embeddings2, labels, temperature, alpha=0.5, device='cpu'):
    """PROPER BioContrast (ICML single-class OCC objective), symmetrised across the two modalities.
    Selected by BC_LOSS_TYPE=biocontrast. BUG2-correct per-anchor label orientation."""
    return biocontrast_occ_cross_modal(embeddings1, embeddings2, labels.T, temperature, device=device) +\
        biocontrast_occ_cross_modal(embeddings2, embeddings1, labels, temperature, device=device)


def nt_bxent_loss_multitask(embeddings1, embeddings2, labels, loss_weights, temperature):
    # Labels expected to be binary
    label_inputs = labels.T
    class_num = labels.size(1)
    if loss_weights is None:
        loss_weights = torch.ones(class_num)
    else:
        assert loss_weights.size(0) == class_num

    for class_id, class_labels in enumerate(label_inputs):
        weight = loss_weights[class_id]
        class_labels = torch.tensor(class_labels, dtype=bool)
    pass


def cosine_loss_mismatch(embedding1, embedding2, temperature):
    logits = (embedding1 @ embedding2.T) / temperature
    embedding1_similarity = embedding1 @ embedding1.T
    embedding2_similarity = embedding2 @ embedding2.T
    targets = F.softmax(
        (embedding1_similarity + embedding2_similarity) / (2*temperature), dim=-1)


def contrastive_loss_cell_line_pdx(cell_line_embedding, pdx_embedding, label, temperature=0.5, alpha=0.5, device='cpu'):
    return nt_bnext_loss(cell_line_embedding, pdx_embedding, label, temperature=temperature, alpha=alpha, device=device)


def binarize_auc_response(auc):
    #breakpoint()
    if type(auc) is pd.Series or type(auc) is pd.DataFrame:
        auc = auc.values
    if type(auc) is torch.Tensor:
        return (auc.clone().detach().requires_grad_(True) < 0.5).int()
    return (torch.tensor(auc) < 0.5).int()


def get_balanced_class_weights(dataset, domain='Response'):
    if domain is None:
        return np.ones(np.shape(dataset)[0])
    weights = torch.empty(dataset.shape[0])
    unique_drugs, counts = np.unique(
        dataset[domain].values, return_counts=True)
    drug_map = {}
    for drug_id, count in zip(unique_drugs, counts):
        drug_map[drug_id] = 1./count

    i = 0
    for drug_id in dataset[domain]:
        weights[i] = drug_map[drug_id]
        i += 1

    return weights


def get_stratified_class_weights(dataset, domain='Response'):
    if domain is None:
        return np.ones(np.shape(dataset)[0])
    weights = torch.empty(dataset.shape[0])
    unique_drugs, counts = np.unique(
        dataset[domain].values, return_counts=True)
    drug_map = {}
    for drug_id, count in zip(unique_drugs, counts):
        drug_map[drug_id] = count

    i = 0
    for drug_id in dataset[domain]:
        weights[i] = drug_map[drug_id]
        i += 1

    return weights
