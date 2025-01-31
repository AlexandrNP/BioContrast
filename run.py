import os
import cProfile
import torch
import torch.nn as nn
from data import ResponseDataloadersFactory, Source, GeneSet, get_pathways
from model import CellLineTransferLearner
from trainer import TransferLearningTrainer
from configuration import Configuration


def run_step(dataloader_factory, configuration, model_tag):
    total_training_steps = configuration['total_training_steps']
    batch_size = 64
    paired_train_dataloader, paired_val_dataloader, paired_test_dataloader = \
        dataloader_factory.get_paired_cell_line_pdx_loaders(
            drug_id, cv_split, num_samples=1000, batch_size=batch_size)
    cell_line_train_loader, cell_line_val_loader, cell_line_test_loader = \
        dataloader_factory.get_drug_specific_cell_line_dataloaders(
            drug_id, cv_split, batch_size=batch_size, num_samples=None)  # num_samples=1000000
    full_dataset_train_cell_line_data_loader, full_dataset_val_cell_line_data_loader, full_dataset_test_cell_line_data_loader = \
        dataloader_factory.get_drug_specific_cell_line_dataloaders(
            drug_id, cv_split, batch_size=8192, num_samples=None)
    pdx_train_loader, pdx_val_loader, pdx_test_loader \
        = dataloader_factory.get_drug_specific_pdx_dataloaders(drug_id, cv_split, batch_size=batch_size, num_samples=10000)

    print(model_tag)
    model = CellLineTransferLearner(configuration=configuration)
    model.to(device)

    train_dataset = pdx_train_loader.dataset
    test_dataset = pdx_test_loader.dataset

    trainer = TransferLearningTrainer(model=model,
                                      tag=model_tag,
                                      steps_per_epoch=1000,
                                      train_matched_data_loader=paired_train_dataloader,
                                      train_cell_line_data_loader=cell_line_train_loader,
                                      full_dataset_train_cell_line_data_loader=full_dataset_train_cell_line_data_loader,
                                      train_pdx_data_loader=pdx_train_loader,
                                      val_matched_data_loader=paired_val_dataloader,
                                      val_cell_line_data_loader=cell_line_val_loader,
                                      val_pdx_data_loader=pdx_val_loader,
                                      test_matched_data_loader=paired_test_dataloader,
                                      test_cell_line_data_loader=cell_line_test_loader,
                                      test_pdx_data_loader=pdx_test_loader,
                                      total_training_steps=total_training_steps,
                                      learning_rate=0.01,
                                      log_interval=10,
                                      save_interval=10,
                                      resume_checkpoint='',
                                      weight_decay=0.05,
                                      learning_rate_anneal_steps=0,
                                      fit_random_forest=False,
                                      l1_regularization_weight=1)

    trainer.train()



if __name__ == "__main__":
    cross_validation_num = 5
    configuration = Configuration()
    device = configuration['device']

    # kegg_pathways = get_pathways()
    # print(list(kegg_pathways[0].keys())[0])

    dataloader_factory = ResponseDataloadersFactory(
        cell_line_source=Source.ALL,
        gene_set=GeneSet.KEGG,
        cross_validation_num=cross_validation_num,
        validation_size=0.12,
        random_seed=2024,
        device=device)

    keys = dataloader_factory.get_paired_dataloaders_keys()
    drugs = [drug_id for drug_id in keys]
    for drug_id in drugs:
        # if drug_id in ['Drug_10']:
        #    continue
        for cv_split in keys[drug_id]:
            model_tag = f'{drug_id}-cvsplit-{cv_split}'
            if os.path.isfile(f'log/best-{model_tag}.pickle'):
                print(f'Skipping {model_tag}...')
                continue
            # cProfile.run('run_step(dataloader_factory, configuration)')
            try:
                run_step(dataloader_factory, configuration, model_tag)
            except:
                continue
