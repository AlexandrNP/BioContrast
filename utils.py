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

    classes = sorted(torch.unique(labels))[-1]
    class_dependent_match = [torch.where(
        labels == y, True, False) for y in classes]

    def compute_loss(mask, inv_mask, logits, axis=-1):
        return torch.mean(torch.stack([
            torch.negative(torch.log(1./sum(mask_i)) + torch.logsumexp(x_i*mask_i, -1)-torch.logsumexp(x_i*inv_mask_i, -1)) if sum(mask_i) > 0 and sum(mask_i) < num_samples else torch.negative(torch.logsumexp(x_i*inv_mask_i, -1)) for x_i, mask_i, inv_mask_i in zip(torch.unbind(logits, dim=axis), torch.unbind(mask, dim=axis), torch.unbind(inv_mask, dim=axis))
        ], dim=axis))

    loss = 0
    for class_labels in class_dependent_match:
        loss += compute_loss(class_labels, ~class_labels, logits)

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
    unique_drugs, counts = torch.unique(dataset[:,2], return_counts=True)
    drug_map = {}
    for drug_id, count in zip(unique_drugs, counts):
        drug_map[int(drug_id.detach().to('cpu'))] = 1./count

    i = 0
    for drug_id in dataset[:,2]:
        weights[i] = drug_map[int(drug_id.detach().to('cpu'))]
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
