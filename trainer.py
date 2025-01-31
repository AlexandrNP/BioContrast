import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from torch import autograd
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, r2_score, confusion_matrix
from utils import binarize_auc_response, l1_norm
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


import pickle
import logger
from data import get_data_generator
from copy import deepcopy

def evaluate_cell_line(evaluations, y_pred_cell_line, y_label_cell_line, dict_entry_name='CellLine'):
    evaluations[dict_entry_name] = {}
    evaluations[dict_entry_name]['MSE'] = mean_squared_error(
        y_label_cell_line, y_pred_cell_line)
    evaluations[dict_entry_name]['R2'] = r2_score(
        y_label_cell_line, y_pred_cell_line)
    evaluations[dict_entry_name]['PCC'] = pearsonr(
        y_label_cell_line, y_pred_cell_line)
    evaluations[dict_entry_name]['SCC'] = spearmanr(
        y_label_cell_line, y_pred_cell_line)
    return evaluations
                
def evaluate_pdx(evaluations, y_pred_pdx, y_label_pdx, dict_entry_name='PDX'):
    evaluations[dict_entry_name] = {}
    binarized_pred_pdx = binarize_auc_response(y_pred_pdx)
    conf_matrix = confusion_matrix(y_label_pdx, binarized_pred_pdx)
    if np.size(conf_matrix) == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
        evaluations[dict_entry_name]['F1'] = f1_score(
            y_label_pdx, binarized_pred_pdx)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        evaluations[dict_entry_name]['F1'] = 0
    if sum(y_label_pdx) < 2:
        auc = 0
    else:
        auc = roc_auc_score(y_label_pdx, y_pred_pdx)
    evaluations[dict_entry_name]['AUC'] = auc  # auc if auc > 0.5 else 1-auc
    evaluations[dict_entry_name]['TPR'] = float(
        tp) / (tp + fn) if (tp+fn) > 1e-3 else 0
    evaluations[dict_entry_name]['TNR'] = float(
        tn) / (tn + fp) if (tn+fp) > 1e-3 else 0
    evaluations[dict_entry_name]['ConfusionMatrix'] = [tn, fp, fn, tp]
    return evaluations

class ModelParser:
    def __init__(self):
        pass

    def parse_resume_step_from_filename(self, filename):
        """
        Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        split = filename.split("model")
        if len(split) < 2:
            return 0
        split1 = split[-1].split(".")[0]
        try:
            return int(split1)
        except ValueError:
            return 0

    def find_resume_checkpoint(self):
        # On your infrastructure, you may want to override this to automatically
        # discover the latest checkpoint on your blob storage, etc.
        return None

    def get_blob_logdir(self):
        dir_path = './log'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        return dir_path


# For now class is implemented without distributed training support


class TransferLearningTrainer:
    def __init__(self,
                 model,
                 tag,
                 train_matched_data_loader,
                 train_cell_line_data_loader,
                 full_dataset_train_cell_line_data_loader,
                 train_pdx_data_loader,
                 val_matched_data_loader,
                 val_cell_line_data_loader,
                 val_pdx_data_loader,
                 test_matched_data_loader,
                 test_cell_line_data_loader,
                 test_pdx_data_loader,
                 steps_per_epoch,
                 total_training_steps,
                 learning_rate,
                 log_interval,
                 save_interval,
                 resume_checkpoint,
                 weight_decay=0.0,
                 learning_rate_anneal_steps=0,
                 l1_regularization_weight=None,
                 fit_random_forest=False):
        self.parser = ModelParser()
        self.model = model #AveragedModel(model)
        self.tag = tag
        self.steps_per_epoch = steps_per_epoch
        self.model_parameters = list(self.model.parameters())
        self.train_cell_line_data_loader = train_cell_line_data_loader
        self.train_cell_line_data_generator = get_data_generator(
            train_cell_line_data_loader)
        self.full_dataset_train_cell_line_data_loader = full_dataset_train_cell_line_data_loader
        self.full_dataset_train_cell_line_data_generator = get_data_generator(
            full_dataset_train_cell_line_data_loader)
        self.train_pdx_data_loader = train_pdx_data_loader
        self.train_pdx_data_generator = get_data_generator(
            train_pdx_data_loader)
        self.train_matched_data_loader = train_matched_data_loader
        self.val_cell_line_data_loader = val_cell_line_data_loader
        self.val_cell_line_data_generator = get_data_generator(
            val_cell_line_data_loader)
        self.val_pdx_data_loader = val_pdx_data_loader
        self.val_pdx_data_generator = get_data_generator(val_pdx_data_loader)
        self.val_matched_data_loader = val_matched_data_loader
        self.val_matched_data_generator = get_data_generator(
            val_matched_data_loader)
        self.test_cell_line_data_loader = test_cell_line_data_loader
        self.test_pdx_data_loader = test_pdx_data_loader
        self.test_matched_data_loader = test_matched_data_loader

        self.total_training_steps = total_training_steps
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.learning_rate_anneal_steps = learning_rate_anneal_steps
        self.contrastive_phase = True
        self.contrastive_loss_min = 1e10
        self.contrastive_epochs = 0
        self.regression_epochs = 0

        self.epoch = 0
        self.resume_epoch = 0
        self.clip_frozen = False

        self.best_model = None # deepcopy(model)
        self._load_and_sync_parameters()
        self.swa_model = None


        self.opt = RAdam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        #self.scheduler = CosineAnnealingLR(self.opt, T_max=50)
        self.swa_start_step = 50
        #self.swa_scheduler = SWALR(self.opt, swa_lr=self.learning_rate)
        if self.resume_epoch:
            self._load_optimizer_state()
        
        self.l1_regularization_weight = l1_regularization_weight

        self.rfr = RandomForestRegressor(n_estimators=10)
        self.fit_random_forest = fit_random_forest
        self.cell_line_regressor_batched_training = True
        self.direct_regressor_batched_training = False
        self.initialized = False

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.parser.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            resume_checkpoint = f'{resume_checkpoint}-{self.tag}'
            self.resume_epoch = self.parser.parse_resume_step_from_filename(
                resume_checkpoint)
            self.model.load_state_dict(resume_checkpoint)

    def _load_optimizer_state(self):
        main_checkpoint = self.parser.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_epoch:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def evaluate(self,
                 matched_data_generator,
                 cell_line_data_generator,
                 pdx_data_generator,
                 comment,
                 print_scores=True
                 ):
        with th.set_grad_enabled(False):
            y_label_cell_line = []
            y_pred_cell_line = []
            y_label_pdx = []
            y_pred_pdx = []
            y_pred_direct_pdx = []
            evaluations = {}

            y_pred_rf = []
            y_pred_direct = []

            self.model.eval()
            for i, (cell_line_rna, auc) in enumerate(cell_line_data_generator):
                if not self.model.direct_only:
                    pred_cell_line = self.model.predict(
                        cell_line_rna, encoding_mode='cell_line', binarize=False)
                    y_pred_cell_line = y_pred_cell_line + pred_cell_line.flatten().tolist()
                y_label_cell_line = y_label_cell_line + auc.flatten().tolist()
                y_pred_direct = y_pred_direct + \
                    self.model.direct_predictor(
                        cell_line_rna).flatten().tolist()
                if self.fit_random_forest:
                    y_pred_rf = y_pred_rf + \
                        list(self.rfr.predict(cell_line_rna.cpu()))

            if not self.model.direct_only:
                for i, (pdx_rna, pdx_response) in enumerate(pdx_data_generator):
                    pred_pdx = self.model.predict(
                        pdx_rna, encoding_mode='pdx', binarize=False)
                    pred_direct_pdx = self.model.predict(pdx_rna, encoding_mode=None, direct=True, binarize=False).cpu().detach().flatten().tolist()
                    y_pred_direct_pdx = y_pred_direct_pdx + pred_direct_pdx 
                    y_pred_pdx = y_pred_pdx + pred_pdx.cpu().detach().flatten().tolist()
                    y_label_pdx = y_label_pdx + pdx_response.cpu().detach().flatten().tolist()

                clip_loss = 0
                for i, full_matched_batch in enumerate(matched_data_generator):
                    matched_batch = full_matched_batch[0:3]
                    # self.model._forward_clip([x.squeeze() for x in matched_batch])
                    clip_loss += self.model._forward_clip(matched_batch)

            #print(clip_loss)
            evaluations['CellLine'] = {}
            if not self.model.direct_only:
                
                evaluations = evaluate_cell_line(evaluations=evaluations, y_pred_cell_line=y_pred_cell_line, y_label_cell_line=y_label_cell_line, dict_entry_name='CellLine')
                evaluations = evaluate_pdx(evaluations=evaluations, y_pred_pdx=y_pred_pdx, y_label_pdx=y_label_pdx, dict_entry_name='PDX')
                evaluations = evaluate_pdx(evaluations=evaluations, y_pred_pdx=y_pred_direct_pdx, y_label_pdx=y_label_pdx, dict_entry_name='Direct_PDX')

                evaluations['CLIP'] = {}
                evaluations['CLIP']['CLIP Loss'] = clip_loss.cpu().detach()

            if self.fit_random_forest:
                evaluations['Random Forest Regressor'] = {}
                evaluations['Random Forest Regressor']['MSE'] = mean_squared_error(
                    y_label_cell_line, y_pred_rf)
                evaluations['Random Forest Regressor']['R2'] = r2_score(
                    y_label_cell_line, y_pred_rf)
                evaluations['Random Forest Regressor']['PCC'] = pearsonr(
                    y_label_cell_line, y_pred_rf)

            evaluations = evaluate_cell_line(evaluations=evaluations, y_pred_cell_line=y_pred_direct, y_label_cell_line=y_label_cell_line, dict_entry_name='MLP')


            def print_all_scores(evaluations, indent='\t'):
                print(
                    f'####### {self.tag} {comment} Epoch {self.epoch} #######')
                for model in evaluations:
                    print(model)
                    for key in evaluations[model]:
                        print(indent, key, evaluations[model][key])
                print('###################################')

            self.model.train()
            if print_scores:
                print_all_scores(evaluations)
            pickle.dump(evaluations, open(
                f'log/{self.tag}-{comment}-{self.epoch}-results.pickle', 'wb'))

        return evaluations

    def train(self):
        saved = False
        best_score = -1e8
        while (self.epoch < self.total_training_steps):

            self.train_cell_line_data_generator = enumerate(
               self.train_cell_line_data_loader)
            self.train_pdx_data_generator = enumerate(
                self.train_pdx_data_loader)
            self.full_dataset_train_cell_line_data_generator = enumerate(
                self.full_dataset_train_cell_line_data_loader)

            full_cell_line_batch = []
            for _, full_cell_line_batch in enumerate(self.full_dataset_train_cell_line_data_loader):
                full_cell_line_batch = full_cell_line_batch + [x.squeeze()
                                        for x in full_cell_line_batch]

            for i, full_matched_batch in enumerate(self.train_matched_data_loader):

                matched_batch = [x.squeeze()
                                 for x in full_matched_batch[0:3]]
                try:
                    cl_idx, cell_line_batch = next(self.train_cell_line_data_generator)
                except:
                    self.train_cell_line_data_generator = enumerate(
                        self.train_cell_line_data_loader)
                    cl_idx, cell_line_batch = next(self.train_cell_line_data_generator)
                pdx_batch = [x.squeeze()
                             for x in [full_matched_batch[1], full_matched_batch[4]]]

                if len(torch.unique(matched_batch[2])) < 3:
                    continue

                if self.fit_random_forest:
                    self.rfr.fit(np.array(cell_line_batch[0].cpu()), np.array(
                        cell_line_batch[1].cpu()))
                    self.rfr.n_estimators += 1

                # with autograd.detect_anomaly():
                self.run_step(matched_batch, cell_line_batch,
                              pdx_batch, full_cell_line_batch)

                if self.epoch % 100 == 0:
                    print_scores = False
                    print(self.epoch)
                    if self.epoch % 100 == 0:
                        print_scores = True

                    if not self.cell_line_regressor_batched_training:
                        self.model.train_regressor(full_cell_line_batch)
                    if not self.direct_regressor_batched_training:
                        self.model._forward_direct_predictor(
                            full_cell_line_batch)
                    val_scores = self.evaluate(self.val_matched_data_loader,
                                               self.val_cell_line_data_loader,
                                               self.val_pdx_data_loader,
                                               comment='Val',
                                               print_scores=False)
                    test_scores = self.evaluate(self.test_matched_data_loader,
                                                self.test_cell_line_data_loader,
                                                self.test_pdx_data_loader,
                                                comment='Test',
                                                print_scores=print_scores)
                    if not self.model.direct_only:
                        f1_val = val_scores['PDX']['F1']
                        auc_val = val_scores['PDX']['AUC']
                        score = auc_val + f1_val
                        if score < 1e-2:
                            score = val_scores['CellLine']['PCC'][0]
                        if print_scores == 0:
                            print(
                                f'Epoch {self.epoch} PDX F1 Validation: {f1_val}')
                            print(
                                f'Epoch {self.epoch} PDX AUC Validation: {auc_val}')
                        if score > best_score:
                            self.best_model = deepcopy(self.model)
                            best_epoch = self.epoch
                            best_score = auc_val 
                            test_scores['best_epoch'] = best_epoch
                            pickle.dump(test_scores, open(
                                f'log/best-{self.tag}.pickle', 'wb'))

                saved = False
                if (
                    self.epoch
                    and self.save_interval != -1
                    and self.epoch % self.save_interval == 0
                ):
                    pass


                if self.epoch % self.log_interval == 0:
                    logger.dumpkvs()

                if i >= self.steps_per_epoch:
                    break
                # print(self.total_training_steps)
                if self.epoch >= self.total_training_steps:
                    break

        if not self.direct_regressor_batched_training:
            self.full_dataset_train_cell_line_data_generator = enumerate(
                self.full_dataset_train_cell_line_data_loader)

            for i, full_cell_line_batch in enumerate(self.full_dataset_train_cell_line_data_loader):
                full_cell_line_batch = [x.squeeze()
                                        for x in full_cell_line_batch]
                self.model.train_regressor(full_cell_line_batch)
                self.model._forward_direct_predictor(
                    full_cell_line_batch)
                test_scores = self.evaluate(self.test_matched_data_loader,
                                            self.test_cell_line_data_loader,
                                            self.test_pdx_data_loader,
                                            comment='Test',
                                            print_scores=True)

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

        return self.model

    def zero_grad(self):
        self._zero_grad(self.model_parameters)

    def _zero_grad(self, model_params):
        for param in model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def _print_norms(self):
        grad_norm, param_norm = self._compute_norms()
        print('Grad norm: ', grad_norm)
        print('Param norm: ', param_norm)

    def _optimize(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
        #print('Grad norm: ', grad_norm)
        #print('Param norm: ', param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.model_parameters:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2,
                                         dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


    def initialize_lazy_layers(self, matched_data, cell_line_data, pdx_data, full_cell_line_data):
        # Initializing lazy layers by running a dummy batch
        init_matched = [torch.rand(x.shape, device=x.device, requires_grad=False) for x in matched_data]
        init_cell_line = [torch.rand(x.shape, device=x.device, requires_grad=False) for x in cell_line_data]
        if pdx_data is None:
            init_pdx_data = pdx_data
        else:
            init_pdx_data = [torch.rand(x.shape, device=x.device, requires_grad=False) for x in pdx_data]
        init_full_cell_line = None
        if full_cell_line_data is not None:
            init_full_cell_line = [torch.rand(x.shape, device=x.device, requires_grad=False) for x in full_cell_line_data]
        #self.forward_backward(init_matched, init_cell_line, init_pdx_data, init_full_cell_line)
        self.model(init_matched, init_cell_line, init_pdx_data, init_full_cell_line)
        self.initialized = True

    def run_step(self, matched_data, cell_line_data, pdx_data, full_cell_line_data):
        freezing_clip = """
        if self.epoch % 2000:
            if self.clip_frozen:
                self.model.clip.freeze()
                self.model.clip.cell_line_encoder.freeze()
                # self.model.cell_line_predictor.freeze()
                self.clip_frozen = True
                # self.model.cell_line_predictor.unfreeze()
            else:
                # self.model.clip.unfreeze()
                # self.model.clip.cell_line_encoder.unfreeze()
                #self.model.clip.freeze()
                #self.model.clip.cell_line_encoder.freeze()
                # self.model.cell_line_predictor.freeze()
                self.clip_frozen = False
                # self.model.cell_line_predictor.freeze()
        #"""
        # if self.clip_frozen:
        #    cell_line_data = None

        # if self.clip_frozen:
        #    self.model.cell_line_predictor.freeze()
        # else:
        #    self.model_parameters
        # cell_line_data = None

        # if self.epoch < 100:
        #    matched_data = None
        # pdx_data = None

        self.zero_grad()
        if not self.initialized:
            self.initialize_lazy_layers(matched_data, cell_line_data,
                              pdx_data, full_cell_line_data)
            #self.swa_model = AveragedModel(self.model)
            return
        self.forward_backward(matched_data, cell_line_data, pdx_data, full_cell_line_data)
        def get_cell_line_loss():
            clip_loss, cell_line_loss, pdx_loss, direct_loss, pdx_direct_loss = self.forward_backward(matched_data, cell_line_data,
                              pdx_data, full_cell_line_data)
            return cell_line_loss
        took_step = None
        #if self.epoch > self.swa_start_step:
        #    took_step = self._optimize(self.swa_scheduler)
        #    self.swa_model.update_parameters(self.model)
        #else:
        #if self.epoch % 30 == 0:
        #    self.scheduler.step()
        #else:
        took_step = self._optimize(self.opt)

        

        if took_step:
            # self.step += 1
            self.epoch += 1

        # self._anneal_lr()
        self.log_step()

    def backward(self, loss: th.Tensor):
        loss.backward()
        # loss.backward(retain_graph=True)

    def forward_backward(self, matched_data, cell_line_data, pdx_data, full_cell_line_data):
        #self.zero_grad()
        #self.opt.zero_grad()
        # if self.contrastive_phase:
        #    print('CONTRASTIVE PHASE')
        # if self.regression_epochs:
        #    print('REGRESSION PHASE')
        clip_loss, cell_line_loss, pdx_loss, direct_loss, pdx_direct_loss = self.model(
            matched_data, cell_line_data, pdx_data, full_cell_line_data)
        if self.epoch % 2 == 0:
            print(f'Cell Line Loss: {cell_line_loss}')
            print(f'CLIP Loss: {clip_loss}')
            print(f'Direct PDX Loss: {pdx_direct_loss}')

        if self.contrastive_phase:
            self.contrastive_epochs += 1
        else:
            self.regression_epochs += 1
        loss_to_accumulate = []

        # print(clip_loss, cell_line_loss, pdx_loss, direct_loss)
        if clip_loss is not None:
            
            #if clip_loss > self.contrastive_loss_min and self.regression_epochs > 300:
            #    self.regression_epochs = 0
            #    self.contrastive_phase = True

            #if self.contrastive_phase:
            #    if clip_loss - self.contrastive_loss_min > 1. and self.contrastive_epochs > 300:
            #        self.contrastive_phase = False
            #        self.contrastive_epochs = 0
            if clip_loss < self.contrastive_loss_min:
                self.contrastive_loss_min = clip_loss
            # print(f'Train CLIP loss: {clip_loss}')
            # loss_to_accumulate.append(clip_loss)
            self.backward(clip_loss)

        if cell_line_loss is not None:
            # print(f'Train cell line loss: {cell_line_loss}')
            # self.backward(cell_line_loss)
            # loss_to_accumulate.append(cell_line_loss)
            # self.model.clip.freeze()

            #if self.l1_regularization_weight is not None:
            #    cell_line_loss += self.l1_regularization_weight*l1_norm(self.model.cell_line_predictor)
            #    cell_line_loss += self.l1_regularization_weight*l1_norm(self.model.clip.cell_line_encoder)
            self.backward(cell_line_loss)
            # self.model.clip.unfreeze()
        if direct_loss is not None:
            self.backward(direct_loss)
        if pdx_loss is not None:
            self.backward(pdx_loss)
        if pdx_direct_loss is not None:
            self.backward(pdx_direct_loss)

        # accumulated_loss = sum(loss_to_accumulate)
        # self.backward(accumulated_loss)
        return clip_loss, cell_line_loss, pdx_loss, direct_loss, pdx_direct_loss

    def save(self):
        import blobfile as bf

        step = self.epoch

        def save_checkpoint(step):
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.opt.state_dict()
            logger.log(f"saving model...")
            filenames = [
                f"model_{self.tag}-{step:08d}.pt", f"optimizer-{step:08d}.pt"]
            param_dicts = [model_state_dict,
                           optimizer_state_dict]
            for base_filename, state_dict in zip(filenames, param_dicts):
                with bf.BlobFile(bf.join(self.parser.get_blob_logdir(), base_filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(step=step)

        logger.log("saving optimizer state...")
        with bf.BlobFile(
            bf.join(self.parser.get_blob_logdir(), f"opt{step:08d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def log_step(self):
        step = self.epoch
        logger.logkv("step", step)
