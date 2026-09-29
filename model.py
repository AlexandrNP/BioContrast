import os
import torch
import lightgbm as lgbm
import torch.nn.functional as F


from torch import nn
from configuration import Configuration, assert_configuration_availability
from modules import MLP, ResNet, BetaRegressionLayer, KEGGHierarchicalCNN, KEGGPathwayBottleneck


def _build_gene_encoder(device):
    """Construct the shared gene->embedding encoder.

    BC_ENCODER selects the encoder for the interpretability ablation:
      'hierarchical' (default) = KEGGHierarchicalCNN, the current behavior.
      'pathway'                = KEGGPathwayBottleneck, the interpretable-by-
                                 construction per-pathway bottleneck whose P
                                 activations ARE the attribution layer.
    Default path is byte-for-byte the original KEGGHierarchicalCNN(device).
    """
    if os.environ.get('BC_ENCODER', 'hierarchical') == 'pathway':
        return KEGGPathwayBottleneck(device)
    return KEGGHierarchicalCNN(device)
from utils import *
from sklearn.model_selection import train_test_split


def _strip_outer_singleton(t):
    """B1 fix: normalize a batch tensor to 2D+ WITHOUT ever removing the batch dim.

    The paired loaders wrap each already-batched tensor in an outer
    ``DataLoader(batch_size=1)`` (data.py), so a matched batch arrives as
    ``[1, B, G]`` / ``[1, B, B]``; the eval/predict path also feeds raw loader
    batches ``[B, G]``. A bare ``.squeeze()`` collapses the batch dim when B==1
    (-> BatchNorm1d "got 1D input" and label "too many indices"). This strips
    ONLY leading size-1 axes while keeping the result at least 2D, so the real
    batch dim survives for every B (including B==1).
    """
    if not torch.is_tensor(t):
        return t
    while t.dim() > 2 and t.size(0) == 1:
        t = t.squeeze(0)
    if t.dim() == 1:
        # A batch that collapsed to 1D upstream (size-1 eval/test fold in small-data CV) must be
        # promoted back to [1, features] -- BatchNorm1d rejects 1D input. In eval mode (predict)
        # a batch of 1 is fine (running stats); in train the matched-batch <2 guard skips it first.
        t = t.unsqueeze(0)
    return t


class ConfigurableFreezableModule(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.device = None
        assert_configuration_availability(module=self,
                                          configuration=configuration)

    def set_device(self, device):
        self.device = device

    def freeze(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self._model.parameters():
            param.requires_grad = True


class ContrastiveProjection(ConfigurableFreezableModule):
    _CONFIG_NAME = "ContrastiveProjection"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        # assert_configuration_availability(module=self,
        #                                  configuration=configuration)
        self._model = MLP(**configuration[self._CONFIG_NAME])

    def forward(self, x):
        return self._model(x)


class CellLineEncoder(ConfigurableFreezableModule):
    _CONFIG_NAME = "CellLineEncoder"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        if 'skip_levels' in configuration[self._CONFIG_NAME]:
            configuration[self._CONFIG_NAME].pop('skip_levels')
        # self._model = MLP(**configuration[self._CONFIG_NAME])
        self._model = _build_gene_encoder(configuration['device'])
        # self._model = ResNet(**configuration[self._CONFIG_NAME])

    def forward(self, x):
        return self._model(x)


class PDXEncoder(ConfigurableFreezableModule):
    _CONFIG_NAME = "PDXEncoder"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        if 'skip_levels' in configuration[self._CONFIG_NAME]:
            configuration[self._CONFIG_NAME].pop('skip_levels')
        self._model = _build_gene_encoder(configuration['device'])
        # self._model = MLP(**configuration[self._CONFIG_NAME])
        # self._model = ResNet(**configuration[self._CONFIG_NAME])

    def forward(self, x):
        return self._model(x)


class DirectPredictor(ConfigurableFreezableModule):
    _CONFIG_NAME = "DirectPredictor"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        # self._model = nn.Sequential(KEGGHierarchicalCNN(configuration['device']),
        #                                nn.LazyLinear(out_features=1),
        #                                nn.Sigmoid())
        self._model = MLP(**configuration[self._CONFIG_NAME])
        #configuration[self._CONFIG_NAME]['output_dim'] = #configuration[self._CONFIG_NAME]['hidden_layers'][-1]
        #self._regressor = BetaRegressionLayer(
        #    configuration[self._CONFIG_NAME]['hidden_layers'][-1])

        #self._model = MLP(**configuration[self._CONFIG_NAME])

    def forward(self, x, y=None):
        if x.size()[0] == 2:
            x = x[0]
        return self._model(x) #self._regressor(self._model(x))

class DirectPDXPredictor(ConfigurableFreezableModule):
    _CONFIG_NAME = "DirectPDXPredictor"

    def __init__(
        self,
        configuration
    ):
            super().__init__(configuration)
            self._model = MLP(**configuration[self._CONFIG_NAME]['DirectPredictor'])

    def forward(self, x):
        return self._model(x)

class ContrastiveLearner(ConfigurableFreezableModule):
    _CONFIG_NAME = "ContrastiveLearner"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        # assert_configuration_availability(module=self,
        #                                  configuration=configuration)
        configuration[self._CONFIG_NAME]['device'] = configuration['device']
        self.cell_line_encoder = CellLineEncoder(
            configuration[self._CONFIG_NAME])
        self.pdx_encoder = PDXEncoder(configuration[self._CONFIG_NAME])
        #self.cell_line_projection = ContrastiveProjection(
        #    configuration[self._CONFIG_NAME])
        #self.pdx_projection = ContrastiveProjection(
        #    configuration[self._CONFIG_NAME])
        self.temperature = configuration[self._CONFIG_NAME]["temperature"]
        # self.alpha = configuration["alpha"]

    def freeze(self):
        self.cell_line_encoder.freeze()
        self.pdx_encoder.freeze()
        #self.cell_line_projection.freeze()
        #self.pdx_projection.freeze()

    def unfreeze(self):
        self.cell_line_encoder.unfreeze()
        self.pdx_encoder.unfreeze()
        #self.cell_line_projection.unfreeze()
        #self.pdx_projection.unfreeze()

    def encode_cell_line(self, data):
        data = self.cell_line_encoder(data)
        return data
        # return self.cell_line_projection(data)

    def encode_pdx(self, data):
        data = self.pdx_encoder(data)
        return data
        # return self.pdx_projection(data)

    def forward(self, batch):
        cell_line_features = self.cell_line_encoder(_strip_outer_singleton(batch[0]))
        pdx_features = self.pdx_encoder(_strip_outer_singleton(batch[1]))
        labels = _strip_outer_singleton(batch[2])
        device = next(self.parameters()).device

        # print("--->PDX encoder:")
        # for name, params in self.pdx_encoder.named_parameters():
        #    print("-->name:",name, "-->max_grad:", params.grad.max(), "-->min_grad:", params.grad.min())
        # print("--->PDX projection:")
        # for name, params in self.pdx_projection.named_parameters():
        #    print("-->name:",name, "-->max_grad:", params.grad.max(), "-->min_grad:", params.grad.min())

        cell_line_embeddings = cell_line_features
        pdx_embeddings = pdx_features

        # cell_line_embeddings = self.cell_line_projection(cell_line_features)
        # pdx_embeddings = self.pdx_projection(pdx_features)

        # loss = contrastive_loss_cell_line_pdx(cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, device=device)

        # LATEST TEST
        # loss = anchorless_sup_con_loss(
        #    cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, device=device)
        # loss = sup_con_mod(cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, device=device)
        # BC_LOSS_TYPE selects the contrastive objective for the loss ablation:
        #   supcon (default) = BioContrast (3-class response-match, multi-positive)
        #   infonce           = single-positive InfoNCE
        #   supcon_std        = standard SupCon (same-response positives, no mismatch clustering)
        #   ntxent            = NT-Xent / CLIP self-supervised (no labels)
        _lt = os.environ.get('BC_LOSS_TYPE', 'supcon')
        if _lt == 'infonce':
            loss = info_nce_transfer_learning(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt == 'supcon_std':
            loss = supcon_std_transfer(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt == 'ntxent':
            loss = ntxent_transfer(cell_line_embeddings, pdx_embeddings, self.temperature)
        elif _lt == 'supcon_fixed':
            loss = supcon_fixed_transfer(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt == 'biocontrast':
            # PROPER BioContrast (ICML): single-class one-class-classification objective (A+U, eqs 4-6).
            loss = biocontrast_transfer(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt == 'supcon_pos':
            loss = supcon_pos_transfer(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt == 'supcon3fix':
            # FIXED 3-class biocontrast loss: cluster the two same-response classes only (mismatch as
            # negatives), restoring a real contrastive gradient (the original 3-class loss is degenerate).
            loss = sup_con_3class_fixed_transfer(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        elif _lt in ('geo', 'geo_aux'):
            # NEGATIVE-MANIFOLD geometry: hypersphere geodesic between cl_i and tg_j proportional to
            # their delta in continuous responsiveness (AUC for PDO/cell-line, ranked for PDX). The
            # trainer forwards the geo Gram (paired-batch idx 5) as batch[3] of the truncated
            # matched_batch; fall back to a neutral zero target if a caller omitted it.
            if len(batch) > 3:
                target_cos = _strip_outer_singleton(batch[3])
            else:
                target_cos = torch.zeros(cell_line_embeddings.size(0), pdx_embeddings.size(0), device=device)
            geo_loss = supcon_geo_transfer(
                cell_line_embeddings, pdx_embeddings, target_cos, temperature=self.temperature, alpha=0.5, device=device)
            if _lt == 'geo_aux':
                # AUXILIARY variant (iter102 next step): keep supcon's discriminative push, add the
                # continuous AUC-ordering as a SOFT regularizer. loss = supcon + lambda * geo.
                lam = float(os.environ.get('BC_GEO_LAMBDA', '0.2'))
                sup_loss = supcon_std_transfer(
                    cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
                loss = sup_loss + lam * geo_loss
            else:
                loss = geo_loss
        else:
            loss = sup_con_transfer_learning_fast(
                cell_line_embeddings, pdx_embeddings, labels, temperature=self.temperature, alpha=0.5, device=device)
        if loss != loss:
            # NaN guard (was a debug breakpoint() in the original). Skip this batch's
            # contrastive contribution rather than halting the whole run.
            loss = torch.zeros((), device=device, requires_grad=True)

        return loss



class LGBMCellLineResponseRegressor(ConfigurableFreezableModule):
    _CONFIG_NAME = "LGBMCellLineResponseRegressor"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        self.device = torch.device(configuration[self._CONFIG_NAME]['device'])
        self.learning_rate = configuration[self._CONFIG_NAME]['learning_rate']
        self.n_estimators = configuration[self._CONFIG_NAME]['n_estimators']
        self.num_leaves = configuration[self._CONFIG_NAME]['num_leaves']
        self.min_split_gain = configuration[self._CONFIG_NAME]['min_split_gain']
        self.reg_alpha = configuration[self._CONFIG_NAME]['reg_alpha']
        self.reg_lambda = configuration[self._CONFIG_NAME]['reg_lambda']
        self.regressor = lgbm.LGBMRegressor(boosting_type='gbdt',
                                            num_leaves=self.num_leaves,
                                            max_depth=-1,
                                            learning_rate=self.learning_rate,
                                            n_estimators=self.n_estimators,
                                            min_split_gain=self.min_split_gain,
                                            min_child_weight=0.001,
                                            min_child_samples=5,
                                            subsample=1.0,
                                            subsample_freq=0,
                                            colsample_bytree=1.0,
                                            reg_alpha=self.reg_alpha,
                                            reg_lambda=self.reg_lambda,
                                            random_state=None,
                                            n_jobs=-4,
                                            importance_type='split')

    def forward(self, x, y=None):
        if y is not None:
            # print(x.size())
            # print(y.size())
            print('Training LGBM regressor...')
            # X = data=x.cpu().detach().numpy()
            # Y = label=y.cpu().detach().numpy()
            # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25)
            # data_all = lgbm.Dataset(X, Y)
            # data_train = lgbm.Dataset(X_train, Y_train)
            # data_val = lgbm.Dataset(X_val, Y_val)
            self.regressor.fit(x.cpu().detach(), y.cpu().detach())
            # params = {}
            # self.regressor = lgbm.train(params, train_set=data_all, num_boost_round=10, init_model = self.regressor)
            # self.regressor = lgbm.train(params, train_set=data_all, valid_sets=None, num_boost_round=300, init_model = None, callbacks=[
            #        #lgbm.early_stopping(stopping_rounds=10),
            #        lgbm.log_evaluation(1)
            # ])
            # self.regressor.fit(x.cpu().detach(), y.cpu().detach())

        return self.predict(x).cpu().detach()

    def predict(self, x):
        x = x.squeeze()
        preds = self.regressor.predict(x.cpu().detach())
        return torch.tensor(preds, requires_grad=True).float().to(self.device)


class CellLineTransferLearner(ConfigurableFreezableModule):
    _CONFIG_NAME = "CellLineTransferLearner"

    def __init__(
            self,
            configuration
    ):
        super().__init__(configuration)
        # assert_configuration_availability(module=self,
        #                                  configuration=configuration)
        self.set_device(configuration['device'])
        configuration[self._CONFIG_NAME]['device'] = self.device
        # BC_HEAD selects the downstream classification-head type for the ablation:
        # 'relu' (default) keeps the non-linear MLP head ([512,256]+ReLU); 'linear'
        # collapses both transfer predictors to a bare Linear+Sigmoid (no hidden layers).
        if os.environ.get('BC_HEAD', 'relu') == 'linear':
            _hcfg = configuration[self._CONFIG_NAME]
            _hcfg['DirectPredictor']['hidden_layers'] = []
            _hcfg['DirectPDXPredictor']['DirectPredictor']['hidden_layers'] = []
        self.clip = ContrastiveLearner(configuration[self._CONFIG_NAME])
        # self.cell_line_predictor = LGBMCellLineResponseRegressor(
        #    configuration[self._CONFIG_NAME])
        self.cell_line_predictor = DirectPredictor(
            configuration[self._CONFIG_NAME])
        self.cell_line_loss_fct = nn.MSELoss()  # nn.BCELoss() #nn.MSELoss()
        self.pdx_loss_fct = nn.BCELoss()
        #self.direct_predictor = LGBMCellLineResponseRegressor(
        #    configuration[self._CONFIG_NAME])
        self.direct_predictor = DirectPredictor(
            configuration[self._CONFIG_NAME])
        self.direct_only = configuration[self._CONFIG_NAME]['direct_only']
        self.pdx_direct_predictor = DirectPDXPredictor(configuration[self._CONFIG_NAME])
        # configuration[self._CONFIG_NAME].pop('direct_only', None)
        # self.direct_predictor = DirectPredictor(configuration[self._CONFIG_NAME])
        self.regressor_batched_training = True

    def freeze(self):
        self.clip.freeze()
        self.cell_line_predictor.freeze()

    def unfreeze(self):
        self.clip.unfreeze()
        self.cell_line_predictor.unfreeze()

    def set_moa_target(self, drug_id=None, pathway_names=None):
        """B2: install the per-drug MoA pathway gate on BOTH KEGG encoders (cell-line +
        target). No-op unless the encoder exposes apply_moa_target (KEGGHierarchicalCNN)
        and BC_MOA_GATE=1 is set at import time. Call once per drug before train/eval;
        pass drug_id=None to clear the gate."""
        n = 0
        for m in self.modules():
            if m is not self and hasattr(m, 'apply_moa_target'):
                m.apply_moa_target(drug_id=drug_id, pathway_names=pathway_names)
                n += 1
        return n

    def _forward_clip(self, matched_batch):
        return self.clip(matched_batch)

    def _forward_cell_line_pred(self, cell_line_batch):
        cell_line_rna = cell_line_batch[0]
        cell_line_response = cell_line_batch[1]
        # print(self.cell_line_predictor.fc1.weight.grad)
        cell_line_embedding = self.clip.encode_cell_line(cell_line_rna)
        cell_line_predictions = self.cell_line_predictor(cell_line_embedding)
        # return self.cell_line_loss_fct(cell_line_response.flatten(), cell_line_predictions.flatten())
        # cell_line_predictions = self.cell_line_predictor(
        #    cell_line_embedding, cell_line_response)
        return self.cell_line_loss_fct(cell_line_response.flatten(), cell_line_predictions.flatten())

    def _forward_direct_predictor(self, cell_line_batch):
        cell_line_rna = cell_line_batch[0]
        cell_line_auc = cell_line_batch[1]

        self.direct_predictor(cell_line_rna, cell_line_auc)
        # self.direct_predictor(cell_line_rna)
        cell_line_predictions = self.direct_predictor(cell_line_rna)
        return self.cell_line_loss_fct(cell_line_auc.flatten(), cell_line_predictions.flatten())

    def _forward_pdx_pred(self, pdx_batch):
        pdx_rna = pdx_batch[0]
        pdx_response = pdx_batch[1]
        pdx_embedding = self.clip.encode_pdx(pdx_rna)
        auc_predictions = self.cell_line_predictor(pdx_embedding).squeeze(-1)
        # binarization_fct = nn.Sigmoid()
        pdx_predictions = auc_predictions  # binarization_fct(auc_predictions)
        return self.pdx_loss_fct(pdx_predictions, pdx_response)
    
    def _forward_direct_pdx_pred(self, pdx_batch):
        pdx_rna = pdx_batch[0]
        pdx_response = pdx_batch[1]
        predictions = self.pdx_direct_predictor(pdx_rna).squeeze(-1)
        #breakpoint()
        #m = nn.Sigmoid()
        return self.pdx_loss_fct(predictions, pdx_response)

    def train_regressor(self, cell_lines_batch):
        cell_line_rna = cell_lines_batch[0]
        cell_line_response = cell_lines_batch[1]
        cell_line_embeddings = self.clip.encode_cell_line(cell_line_rna)
        self.cell_line_predictor(cell_line_embeddings, cell_line_response)

    def predict(self, X, encoding_mode='cell_line', direct=False, binarize=False):
        X = _strip_outer_singleton(X)
        embeddings = None
        if encoding_mode == 'cell_line':
            embeddings = self.clip.encode_cell_line(X)
        elif encoding_mode == 'pdx':
            embeddings = self.clip.encode_pdx(X)
        elif encoding_mode == None and direct:
            embeddings = X
        else:
            raise Exception(f'Unknown encoding mode: {encoding_mode}')

        if direct:
            predictions = self.pdx_direct_predictor(embeddings)
            if binarize:
                predictions = binarize_auc_response(predictions)
            return predictions

        predictions = self.cell_line_predictor(embeddings)
        if binarize:
            predictions = binarize_auc_response(predictions)

        return predictions

    def forward(self, matched_batch, cell_line_batch, pdx_batch, full_cell_line_data=None):
        clip_loss = None
        cell_line_loss = None
        pdx_loss = None
        direct_predictor_loss = None
        pdx_direct_loss = None

        #matched_batch = None
        #pdx_batch = None

        if self.direct_only:
            direct_predictor_loss = self._forward_direct_predictor(
                cell_line_batch)
            return None, None, None, direct_predictor_loss
        else:
            direct_predictor_loss = self._forward_direct_predictor(
                cell_line_batch)

        if matched_batch is not None:
            any_nans = 0
            for parameter in self.clip.cell_line_encoder.parameters():
                any_nans += torch.isnan(parameter).sum()
            if any_nans > 0:
                pass  # was a debug breakpoint()
            clip_loss = self._forward_clip(matched_batch)
        if cell_line_batch is not None: #and self.regressor_batched_training:
            cell_line_loss = self._forward_cell_line_pred(cell_line_batch)
            # if full_cell_line_data is None:
            #    direct_predictor_loss = self._forward_direct_predictor(
            #        cell_line_batch)
            # else:
            #    direct_predictor_loss = self._forward_direct_predictor(
            #        full_cell_line_data)
        if pdx_batch is not None:
            pdx_loss = self._forward_pdx_pred(pdx_batch)
            pdx_direct_loss = self._forward_direct_pdx_pred(pdx_batch)

        return clip_loss, cell_line_loss, pdx_loss, direct_predictor_loss, pdx_direct_loss
