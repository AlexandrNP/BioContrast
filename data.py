import os
import sys
import pickle
import numpy as np
import pandas as pd
from enum import Enum
from copy import deepcopy
from Bio.KEGG.REST import *
from utils import binarize_auc_response, get_balanced_class_weights, get_stratified_class_weights
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, ShuffleSplit
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate


import pickle
pickle.HIGHEST_PROTOCOL = 4

KEGG_FILE = 'KEGG/KEGG.pickle'
KEGG_DIR = 'KEGG'
KEGGID_DIR = os.path.join(KEGG_DIR, 'KEGGID')
KEGG_GENE_DIR = os.path.join(KEGG_DIR, 'Symbol')

DATA_DIR = os.path.join('pdo_data', 'raw_data')
CELL_LINE_DIR = os.path.join(
    DATA_DIR)
CELL_LINE_EXPRESSION_DIR = os.path.join(CELL_LINE_DIR, 'x_data')
CELL_LINE_RESPONSE_DIR = os.path.join(CELL_LINE_DIR, 'y_data')
DRUG_DIR = CELL_LINE_EXPRESSION_DIR #os.path.join(CELL_LINE_DIR, 'Drug_Data')
#NOVARTIS_DIR = os.path.join(CELL_LINE_DIR, 'PDX_Data')
NIH_DIR = CELL_LINE_EXPRESSION_DIR
PROCESSED_DIR = os.path.join(DATA_DIR, 'Processed')
ENCODED_DATASET_PATH = os.path.join(PROCESSED_DIR, 'encoded_dataset.pickle')


class Source(Enum):
    ALL = 'ALL'
    CCLE = 'CCLE'
    CTRP = 'CTRPv2'
    GDSC = 'GDSCv2'
    gCSI = 'gCSI'
    NCI60 = 'NCI60'


class GeneSet(Enum):
    ALL = os.path.join(
        DATA_DIR, 'GeneSets', 'all.txt')
    LINCS = os.path.join(DATA_DIR, 'GeneSets', 'lincs1000_list.txt')
    ONCOGENES_REG = os.path.join(
        DATA_DIR, 'GeneSets', 'oncogenes_list.txt')
    ONCOGENES_DOCKING = os.path.join(
        DATA_DIR, 'GeneSets', 'oncogenes_gausschem4.txt')
    KEGG = os.path.join(
        DATA_DIR, 'GeneSets', 'KEGG.txt')


class DrugInfo(Enum):
    SMILES = 'SMILES'
    DESCRIPTORS = 'descriptors'
    ECFP = 'ECFP'
    PFP = 'PFP'


def get_gene_list(gene_set):
    # if gene_set == GeneSet.ALL:
    #    return None
    gene_list = list(pd.read_csv(
        gene_set.value, header=None).transpose().values[0]) + ['Sample']
    return gene_list


def get_metadata():
    path = os.path.join(CELL_LINE_EXPRESSION_DIR, 'combined_metadata_pdo.tsv')
    data = pd.read_csv(path, sep='\t')
    data = data[['sample_name', 'simplified_tumor_site', 'simplified_tumor_type']]
    data.columns = ['Sample', 'simplified_tumor_site', 'simplified_tumor_type']
    return data


def pickle_load(pickle_path):
    def decorator(load_function):
        def wrapper(*args, **kwargs):
            suffix = ''
            if 'source' in kwargs:
                source = kwargs['source'].value
                suffix = f'_{source}'
            if 'gene_set' in kwargs:
                gene_set = kwargs['gene_set'].value.split(
                    '/')[-1].split('.')[0]
                suffix = f'{suffix}_{gene_set}'
            pickle_path_new = f'{pickle_path}{suffix}.pickle'
            if os.path.exists(pickle_path_new):
                return pickle.load(open(pickle_path_new, 'rb'))
            data = load_function(*args, **kwargs)
            pickle.dump(data, open(pickle_path_new, 'wb'))
            return data
        return wrapper
    return decorator


def csv_load(csv_path, sep='\t'):
    def decorator(load_function):
        def wrapper(*args, **kwargs):
            suffix = ''
            if 'source' in kwargs:
                source = kwargs['source'].value
                suffix = f'_{source}'
            if 'gene_set' in kwargs:
                gene_set = kwargs['gene_set'].value.split(
                    '/')[-1].split('.')[0]
                suffix = f'{suffix}_{gene_set}'
            pickle_path_new = f'{csv_path}{suffix}.csv'
            if os.path.exists(pickle_path_new):
                return pd.read(csv_path, sep=sep)
            data = load_function(*args, **kwargs)
            data.to_csv(pickle_path_new, sep=sep)
            return data
        return wrapper
    return decorator


def hdf5_load(hdf5_path, sep='\t'):
    def decorator(load_function):
        def wrapper(*args, **kwargs):
            suffix = ''
            if 'source' in kwargs:
                source = kwargs['source'].value
                suffix = f'_{source}'
            if 'gene_set' in kwargs:
                gene_set = kwargs['gene_set'].value.split(
                    '/')[-1].split('.')[0]
                suffix = f'{suffix}_{gene_set}'
            pickle_path_new = f'{hdf5_path}{suffix}.h5'
            pickle_path_supplemental = f'{hdf5_path}{suffix}_supplemental.pickle'
            if os.path.exists(pickle_path_new):
                data = pd.read_hdf(pickle_path_new)

                if os.path.exists(pickle_path_supplemental):
                    supplemental = pickle.load(
                        open(pickle_path_supplemental, 'rb'))
                    return (data,) + supplemental

                return data

            if suffix == '':
                suffix = '_df'
            info = load_function(*args, **kwargs)
            if type(info) == tuple:
                data = info[0]
                supplemental = info[1:]
                pickle.dump(supplemental, open(pickle_path_supplemental, 'wb'))
                data.to_hdf(pickle_path_new, key=suffix[1:])
                return (data,) + supplemental

            data = info
            data.to_hdf(pickle_path_new, key=suffix[1:])
            return data

        return wrapper
    return decorator


def get_drug_data(data_type=DrugInfo.SMILES):
    drug_file_descriptors = 'JasonPanDrugsAndNCI60_dragon7_descriptors.tsv'
    drug_file_ECFP = 'JasonPanDrugsAndNCI60_dragon7_ECFP.tsv'
    drug_file_PFP = 'JasonPanDrugsAndNCI60_dragon7_PFP.tsv'
    drug_file_smiles = 'drug_SMILES_combined.tsv'

    drug_file = None
    if data_type == DrugInfo.DESCRIPTORS:
        drug_file = drug_file_descriptors
    elif data_type == DrugInfo.ECFP:
        drug_file = drug_file_ECFP
    elif data_type == DrugInfo.PFP:
        drug_file = drug_file_PFP
    elif data_type == DrugInfo.SMILES:
        drug_file = drug_file_smiles
    else:
        raise Exception(
            f'Unknown drug format {data_type}. The supported types are \'descriptors\', \'ECFP\', \'PFP\', \'SMILES\'.')

    drug_path = os.path.join(DRUG_DIR, drug_file)
    drug_data = pd.read_csv(drug_path, sep='\t')
    if data_type == DrugInfo.SMILES:
        drug_data['UniqueID'] = drug_data['improve_chem_id']
        drug_data['SMILES(drug_info)'] = drug_data['canSMILES']
        drug_data = drug_data.set_index('UniqueID', drop=False)
        drug_data = drug_data[[
            'UniqueID', 'SMILES(drug_info)']]
        drug_data = drug_data.dropna(how='all')

        drug_data = drug_data.transpose()
        drug_data.columns = drug_data.loc['UniqueID', :]
        drug_data = drug_data.drop('UniqueID')
        drug_data['smiles'] = np.repeat('smiles', drug_data.shape[0])
        drug_data = drug_data.groupby('smiles').first()
        drug_data = drug_data.transpose()
        drug_data['UniqueID'] = drug_data.index
        drug_data = drug_data.reset_index(drop=True)

    return drug_data


def load_gene_expression_data(datapath, gene_set, separator='\t'):
    data = None
    if gene_set == GeneSet.ALL:
        print('ALL GENES RECORDERED')
        data = pd.read_csv(datapath, sep=separator)
    else:
        print('GENE LIST ENACTED')
        gene_list = get_gene_list(gene_set)
        data = pd.read_csv(datapath, sep=separator,
                           usecols=lambda x: x in gene_list)
        #gene_list = [x for x in gene_list if x != 'T']
        data = data[gene_list]
    return data


def map_nci_drug_id(target_pdx_df):
    drug_map = {}
    drug_info_all = pd.read_csv(os.path.join(
        DRUG_DIR, 'Drugs_For_OV_Proposal_Analysis.txt'), sep='\t')
    #drug_info_all = drug_info_all[[
    #    'UniqueID', 'NSC', 'NSC.ID(NCI_IOA_AOA_drugs)', 'NSC.ID(NCI60_drug)']]
    for i in range(drug_info_all.shape[0]):
        idx = drug_info_all.index[i]
        improve_chem_id = drug_info_all.loc[idx, 'improve_chem_id']
        drug_name = drug_info_all.loc[idx, 'drug_name']
        drug_map[improve_chem_id] = drug_name
        #for col in ['NSC', 'NSC.ID(NCI_IOA_AOA_drugs)', 'NSC.ID(NCI60_drug)']:
        #    nsc_id = str(drug_info_all.loc[idx, col]).split('.')[-1]
        #    drug_map[nsc_id] = unique_id

    to_drop = [idx for idx in target_pdx_df.index if target_pdx_df.loc[idx,
                                                                       'Drug'].split('.')[-1] not in drug_map]
    drugs_to_drop = np.unique(target_pdx_df.loc[to_drop, 'Drug'])
    target_pdx_df = target_pdx_df.drop(to_drop, axis=0)
    target_pdx_df['Drug'] = [drug_map[str(
        x.split('.')[-1])] if 'NSC' in x else x for x in target_pdx_df['Drug']]
    return target_pdx_df


@hdf5_load(os.path.join(PROCESSED_DIR, 'celllines_gene_expressions'))
def get_cell_line_expression(gene_set=GeneSet.ALL):
    datapath = os.path.join(CELL_LINE_EXPRESSION_DIR, 'cancer_gene_expression_curated.tsv')
    data = load_gene_expression_data(datapath, gene_set)
    data.set_index('Sample', drop=False)
    data = data[~data.index.duplicated(keep='first')]
    return data


@hdf5_load(os.path.join(PROCESSED_DIR, 'novartis_pdx_gene_expressions'))
def get_novartis_pdx_expression(gene_set=GeneSet.ALL):
    datapath = os.path.join(CELL_LINE_EXPRESSION_DIR,
                            'combined_rnaseq_data_novartis')
    data = load_gene_expression_data(datapath, gene_set).fillna(0)
    return data


@hdf5_load(os.path.join(PROCESSED_DIR, 'nih_pdx_gene_expressions'))
def get_nih_pdx_expression(gene_set=GeneSet.ALL):
    datapath = os.path.join(NIH_DIR, 'cancer_gene_expression_curated.tsv')
    data = load_gene_expression_data(datapath, gene_set)
    pdx_ids = [x for x in data['Sample'] if('_' in str(x))] #or 'CO-' in str(x) or 'HN-' in str(x) or 'CR-' in str(x))]
    data = data.set_index('Sample', drop=False)
    data = data[~data.index.duplicated(keep='first')]
    data = data.fillna(0)
    return data.loc[pdx_ids]


def get_cell_line_response(source=Source.ALL):
    data = pd.read_csv(os.path.join(CELL_LINE_RESPONSE_DIR,
                       'response.tsv'), sep='\t', low_memory=False)
    if source != Source.ALL:
        data = data.loc[data.index[data['source'] == source.value]][[
            'improve_sample_id', 'improve_chem_id', 'auc']]
    else:
        data = data[['improve_sample_id', 'improve_chem_id', 'auc']]
    data['Sample'] = data['improve_sample_id']
    data['Drug'] = data['improve_chem_id']
    data['Response'] = data['auc']
    data = data[['Sample', 'Drug', 'Response']]
    return data


def get_novartis_pdx_response():
    path = os.path.join(
        NOVARTIS_DIR, 'ncipdm_novartis_single_drug_response.txt')
    data = pd.read_csv(path, sep='\t')
    data = data.loc[data.index[data['Source'] == 'Novartis']]
    
    return data[['Sample', 'Drug', 'Response']]


def get_nih_pdx_response():
    path = os.path.join(
        NIH_DIR, 'PDO_response_combined_v2.tsv')
    # 'ncipdm_drug_response_preprocessed_rare_Oct_2023.tsv')

    data = pd.read_csv(path, sep='\t')
    
    data['Sample'] = data['Organoid']
    data['Drug'] = data['Drug']
    data['Response'] = data['AUC'] < 0.5

    return data[['Sample', 'Drug', 'Response']]


def get_novartis_nih_pdx_response():
    # data = pd.read_csv(os.path.join(
    #    NIH_DIR, 'ncipdm_drug_response_preprocessed.tsv'), sep='\t')
    path = os.path.join(
        NOVARTIS_DIR, 'cancer_gene_expression_curated.tsv')

    data = pd.read_csv(path, sep='\t')
    data = data.loc[data.index[data['Source'] == 'NCIPDM']]
    # d = pd.read_csv(
    #    data_config.preclinical_pdx_drug_response_file, sep='\t')
    pdx_df = data[['Sample', 'Drug', 'Response']]
    pdx_df['Response'] = [
        0 if x > 0.5 else 1 for x in pdx_df['Response']]
    pdx_df.dropna(inplace=True)
    pdx_df['Groups'] = ['-'.join(sample.split('~')[:-1])
                        for sample in pdx_df['Sample']]
    pdx_df = pdx_df.groupby(by='Groups').first()

    return data[['Sample', 'Drug', 'Response']]


@hdf5_load(os.path.join(PROCESSED_DIR, 'cellline_dataset'))
def get_cell_line_dataset(source, gene_set):
    drugs_data = get_drug_data()
    cell_line_rna = get_cell_line_expression(gene_set=gene_set)
    cell_line_response = get_cell_line_response(source)
    metadata = get_metadata()
    id_columns = ['Sample', 'UniqueID']
    drop_columns = ['Drug_UniqueID', 'CELL']
    drug_columns = [
        x for x in drugs_data.columns if x not in id_columns and x not in drop_columns]
    rna_columns = [
        x for x in cell_line_rna.columns if x not in id_columns and x not in drop_columns]
    targets_columns = [
        x for x in cell_line_response.columns if x not in id_columns and x not in drop_columns]
    metadata_columns = [
        x for x in metadata.columns if x not in id_columns and x not in drop_columns]

    rna_response = cell_line_rna.merge(
        cell_line_response, left_on='Sample', right_on='CELL')
    cell_line_dataset = rna_response.merge(
        drugs_data, left_on='Drug_UniqueID', right_on='UniqueID')
    cell_line_dataset = cell_line_dataset.merge(metadata, on='Sample')
    cell_line_dataset.drop(drop_columns, axis=1, inplace=True)

    return cell_line_dataset, id_columns, rna_columns, drug_columns, metadata_columns, targets_columns


# @hdf5_load(os.path.join(PROCESSED_DIR, 'novartis_dataset'))
def get_novartis_dataset(gene_set):
    drugs_data = get_drug_data()
    novartis_rna = get_novartis_pdx_expression(gene_set=gene_set)
    novartis_response = get_novartis_pdx_response()
    # print(novartis_response)
    novartis_response = map_nci_drug_id(novartis_response)
    # print(novartis_response)
    metadata = get_metadata()
    id_columns = ['Sample', 'UniqueID']
    drop_columns = ['Drug']
    drug_columns = [
        x for x in drugs_data.columns if x not in id_columns and x not in drop_columns]
    rna_columns = [
        x for x in novartis_rna.columns if x not in id_columns and x not in drop_columns]
    targets_columns = [
        x for x in novartis_response.columns if x not in id_columns and x not in drop_columns]
    metadata_columns = [
        x for x in metadata.columns if x not in id_columns and x not in drop_columns]

    rna_response = novartis_rna.merge(
        novartis_response, left_on='Sample', right_on='Sample')
    novartis_dataset = rna_response.merge(
        drugs_data, left_on='Drug', right_on='UniqueID')
    novartis_dataset = novartis_dataset.merge(metadata, on='Sample')
    return novartis_dataset, id_columns, rna_columns, drug_columns, metadata_columns, targets_columns


@hdf5_load(os.path.join(PROCESSED_DIR, 'nih_dataset_smiles'))
def get_nih_dataset(gene_set):
    drug_data = get_drug_data()
    nih_rna = get_nih_pdx_expression(gene_set=gene_set)
    nih_response = get_nih_pdx_response()
    nih_response = map_nci_drug_id(nih_response)
    metadata = get_metadata()
    id_columns = ['Sample', 'UniqueID']
    drop_columns = ['Drug']
    drug_columns = [
        x for x in drug_data.columns if x not in id_columns and x not in drop_columns]
    rna_columns = [
        x for x in nih_rna.columns if x not in id_columns and x not in drop_columns]
    targets_columns = [
        x for x in nih_response.columns if x not in id_columns and x not in drop_columns]
    metadata_columns = [
        x for x in metadata.columns if x not in id_columns and x not in drop_columns]

    rna_response = nih_rna.merge(
        nih_response, left_on='Sample', right_on='Sample')
    nih_dataset = rna_response.merge(
        drug_data, left_on='Drug', right_on='UniqueID')
    nih_dataset = nih_dataset.merge(metadata, on='Sample')

    return nih_dataset, id_columns, rna_columns, drug_columns, metadata_columns, targets_columns

# This class automatically constructs output based on drug response pair-input dataset.
# RNA-Seq data and drug descriptors are stored in separate datasets without merging.
# response_df contain single response metric columns with multiindex.
# This class assumes that rna_df indices contain sample names that correspond to top-level index in response_df
# and drug_df indices correspond to second-level index in response_df multiindex.


class LookupDRPDataLoader(data.Dataset):
    class MODE(Enum):
        RNA_ITERATOR = 0,
        DRUG_ITERATOR = 1,
        RESPONSE_ITERATOR = 2

    def _serialize(self):
        pickle.dump(self, open(self.serialization_path, 'wb'))

    def __init__(self,
                 mode,
                 rna_df,
                 drug_df,
                 response_df,
                 rna_metadata_df=None,
                 drug_metadata_df=None,
                 add_noise=False,
                 device=None):
        self._encoder_dicts = None
        self.serialization_path = os.path.join(
            '.', 'LookupDRPDataLoader.pickle')
        # drug_metadata_df = drug_metadata_df.set_index(
        #    self.drug_label_encoder.fit_transform(drug_metadata_df.index))

        self.device = device
        self.mode = mode
        self.rna_df = rna_df
        self.drug_df = drug_df
        self.rna_metadata_df = rna_metadata_df
        self.drug_metadata_df = drug_metadata_df
        self.response_df = response_df
        self.add_noise = add_noise
        self._serialize()

    def get_rna_dim(self):
        return self.rna_df.shape[1]

    def get_drug_dim(self):
        return self.drug_df.shape[1]

    def __len__(self):
        if self.mode == self.MODE.DRUG_ITERATOR:
            return self.drug_df.shape[0]
        if self.mode == self.MODE.RNA_ITERATOR:
            return self.rna_df.shape[0]
        if self.mode == self.MODE.RESPONSE_ITERATOR:
            return self.response_df.shape[0]
        raise Exception('Unknown running mode')

    def __getitem__(self, index):
        rna_idx = None
        drug_idx = None
        rna_sample = None
        drug_sample = None
        label = None
        if self.mode == self.MODE.DRUG_ITERATOR:
            return self.drug_df.loc[self.drug_df.index[index]]
        if self.mode == self.MODE.RNA_ITERATOR:
            rna_idx = self.rna_df.index[index]
            rna_sample = self.rna_df.loc[rna_idx]
        if self.mode == self.MODE.RESPONSE_ITERATOR:
            rna_idx, drug_idx = self.response_df.index[index]
            rna_sample = self.rna_df.loc[rna_idx]
            drug_sample = self.drug_df.loc[drug_idx]
            label = self.response_df.loc[self.response_df.index[index]]

        if self.add_noise:
            std = 0.05
            rna_sample = rna_sample + \
                np.random.normal(0, std, size=len(
                    self.rna_diameters))*self.rna_diameters
            rna_sample = np.array(rna_sample)

        rna_metadata = None
        drug_metadata = None
        if self.rna_metadata_df is not None:
            rna_metadata = self.rna_metadata_df.loc[rna_idx].values
        if self.drug_metadata_df is not None:
            if drug_idx is None:
                raise Exception(
                    "A wrong running mode! There is no drug information but drug metadata is present.")
            drug_metadata = self.drug_metadata_df.loc[drug_idx].values
        if self.device is not None:
            if rna_sample is None:
                rna_sample = 0
            if drug_sample is None:
                drug_sample = 0
            if label is None:
                label = 0
            if rna_metadata is None:
                rna_metadata = 0
            if drug_metadata is None:
                drug_metadata = 0
            rna_sample = Variable(torch.from_numpy(
                np.array(rna_sample, dtype=np.float16))).float().to(self.device)

            drug_sample = Variable(torch.from_numpy(
                np.array(drug_sample, dtype=np.float16))).float().to(self.device)

            label = Variable(torch.from_numpy(
                np.array(label))).float().to(self.device)

        return rna_sample, drug_sample, label, rna_metadata, drug_metadata


class DRPDataLoader(data.Dataset):
    def __init__(self,
                 device,
                 indices,
                 rna_df,
                 drug_df,
                 metadata_df,
                 labels,
                 add_noise=False):
        # list_IDs, labels, drug_df, rna_df, binding_df, add_noise=False):
        'Initializing...'
        self.device = device
        drug_df = drug_df.reset_index()
        self.labels = labels
        self.indices = indices
        self.drug_df = pd.DataFrame(drug_df).reset_index(drop=True)
        self.rna_df = pd.DataFrame(rna_df).reset_index(drop=True)
        self.metadata_df = pd.DataFrame(metadata_df).reset_index(drop=True)

        self.drug_df.loc[drug_df.index, 'UniqueID'] = [
            int(x.split('_')[-1]) for x in drug_df['UniqueID']]

        self.drug_ids = torch.tensor(self.drug_df['UniqueID'])
        # self.drug_diameters = drug_df.apply(np.ptp, axis=0)
        # self.drug_size = drug_df.shape[1]
        self.rna_size = rna_df.shape[1]
        self.add_noise = add_noise

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.indices[index]
        v_d = self.drug_df.iloc[index]  # ['drug_encoding']
        v_p = np.array(self.rna_df.iloc[index])
        # v_b = np.array(self.binding_df.iloc[index])
        d_id = np.array(self.drug_df.iloc[index]['UniqueID'])
        if self.add_noise:
            std = 0.05
            # v_d = v_d + np.random.multivariate_normal(np.repeat(0, self.drug_size), np.diag(self.drug_diameters*std))
            # np.random.multivariate_normal(np.repeat(0, self.rna_size), np.diag(self.rna_diameters*std))
            v_p = v_p + \
                np.random.normal(0, std, size=len(
                    self.rna_diameters))*self.rna_diameters
            v_p = np.array(v_p)
            # np.random.multivariate_normal(np.repeat(0, self.binding_size), np.diag(self.binding_diameters*std))
            v_b = v_b + \
                np.random.normal(0, std, size=len(
                    self.binding_diameters))*self.binding_diameters
            v_b = np.array(v_b)
        y = np.array(self.labels[index])

        return v_d, v_p, v_b, y, d_id


class ContrastiveDrugCellLineDataset(data.Dataset):
    def __init__(self,
                 device,
                 indices,
                 cell_line_rna_df,
                 drug_df,
                 metadata_df,
                 labels,
                 add_noise=False):
        self.device = device
        drug_df = drug_df.reset_index()
        self.labels = labels
        self.indices = indices
        self.drug_df = pd.DataFrame(drug_df).reset_index(drop=True)
        self.cell_line_rna_df = pd.DataFrame(
            cell_line_rna_df).reset_index(drop=True)
        self.metadata_df = pd.DataFrame(metadata_df).reset_index(drop=True)


def get_drug_cell_line_paired_loader(source, gene_set):
    cell_line_dataset, id_columns, rna_columns, drug_columns, metadata_columns, targets_columns = get_cell_line_dataset(
        source, gene_set)


def get_data():
    cell_data, cell_id_columns, cell_rna_columns, cell_drug_columns, cell_metadata_columns, cell_targets_columns = get_cell_line_dataset(
        source=Source.ALL, gene_set=GeneSet.ALL)

    cell_to_drop = ['SOURCE', 'CCLE_CCL_UniqueID', 'NCI60_CCL_UniqueID', 'DRUG', 'STUDY',
                    'AUC1', 'AAC1', 'DSS1', 'IC50', 'EC50', 'EC50se', 'R2fit', 'Einf', 'HS']  # 'AUC',
    cell_data.drop(cell_to_drop, axis=1, inplace=True)

    pdx_data, pdx_id_columns, pdx_rna_columns, pdx_drug_columns, pdx_metadata_columns, pdx_targets_columns = get_nih_dataset(
        gene_set=GeneSet.ALL)


def expand_nih_response_dataset(nih_response_df, individual_samples):
    corresponding_families = ['-'.join(x.split('-')[:-1])
                              for x in individual_samples]
    families_df = pd.DataFrame(np.array([individual_samples, corresponding_families]).transpose(
    ), columns=['IndividualSample', 'Sample'])
    expanded_df = nih_response_df.merge(families_df, how='inner', on='Sample')
    expanded_df.drop('Sample', axis=1, inplace=True)
    expanded_df.columns = ['Sample' if x ==
                           'IndividualSample' else x for x in expanded_df.columns]

    return expanded_df


class ResponseDataset(Dataset):
    def __init__(self, response_df, rna_df, device, sample_indices=None):
        super().__init__()
        if type(rna_df) is not torch.Tensor:
            self.sample_indices = torch.tensor(rna_df.index.values).int().to(device)
            self.rna_df = torch.tensor(rna_df.values).to(device)
            #if self.rna_indices is None:
            #    self.rna_indices = torch.tensor(rna_df.index.values).to(device)
        else:
            self.rna_df = rna_df
            self.sample_indices = sample_indices.int()
        if type(response_df) is not torch.Tensor:
            response_df.reset_index(inplace=True)
            self.response_indices = torch.tensor(response_df.index.values).to(device)
            self.response_df = torch.tensor(response_df.values).to(device)
            #if self.response_indices is None:
            #    self.response_indices = torch.tensor(response_indices.index.values).to(device)
        else:
            self.response_df = response_df
            self.response_indices = torch.tensor(list(range(self.response_df.shape[0]))).int().to(device)
        #self.indices = list(range(self.response_df.shape[0])) #self.response_df #.index
        self.device = device

    def __getitem__(self, index):
        sample_id = torch.nonzero(self.sample_indices == self.response_df[self.response_indices[index],1], as_tuple=True)[0][0]
        return self.rna_df[sample_id.item(),:].float(), \
            self.response_df[self.response_indices[index].item(),2].float()

    def __len__(self):
        return self.response_df.shape[0]


class DrugSpecificPairedDataset(Dataset):
    def _serialize(self):
        pickle.dump(self, open(self.serialization_path, 'wb'))

    #
    # Paired dataset would return a tuple (Data1_batch, Data2_batch, match_matrix) of the size (batch_size, batch_size, batch_size*batch_size)
    # Match is counted when cell line and pdx model both result in response
    #
    # num_samples: number of samples that paired dataset will generate
    # batch_size: number of samples returned after each dataset query
    #

    def __init__(self,
                 num_samples,
                 batch_size,
                 cell_line_rna_df,
                 cell_line_response_df,
                 pdx_rna_df,
                 pdx_response_df,
                 cell_line_indices=None,
                 pdx_indices=None,
                 cell_line_rna_metadata_df=None,
                 pdx_rna_metadata_df=None,
                 response_match_function=None,
                 device=None,
                 serialize=False):
        super().__init__()

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.cell_line_indices = cell_line_indices
        self.pdx_indices = pdx_indices

        self._encoder_dicts = None
        self.serialization_path = os.path.join(
            '.', 'PairedDataset.pickle')
        # drug_metadata_df = drug_metadata_df.set_index(
        #    self.drug_label_encoder.fit_transform(drug_metadata_df.index))

        self.device = device

        self.cell_line_rna_df = cell_line_rna_df
        self.cell_line_response_df = cell_line_response_df
        self.pdx_rna_df = pdx_rna_df
        self.pdx_response_df = pdx_response_df

        self.cell_line_rna_metadata_df = cell_line_rna_metadata_df
        self.pdx_rna_metadata_df = pdx_rna_metadata_df

        unique_cell_line_samples = torch.unique(
            self.cell_line_response_df[:,1]) # 'Sample' columns
        cell_line_dataset = ResponseDataset(self.cell_line_response_df,
                                            self.cell_line_rna_df.loc[unique_cell_line_samples.detach().to('cpu')],
                                            sample_indices=self.cell_line_indices,
                                            device=self.device)
        unique_pdx_samples = torch.unique(self.pdx_response_df[:,1]) # 'Sample' columns
        pdx_dataset = ResponseDataset(self.pdx_response_df,
                                      self.pdx_rna_df.loc[unique_pdx_samples.detach().to('cpu')],
                                      sample_indices=self.pdx_indices,
                                      device=self.device)

        # self.matches = pd.DataFrame(unique_cell_line_samples.transpose()).merge(pd.DataFrame(unique_pdx_samples.transpose()), how='cross')
        # self.matches.columns = ['CellLine_Sample', 'PDX_Sample']
        self.batches = {}

        def create_dataset_balanced(self, response_match_function):
            cell_line_weights = get_balanced_class_weights(
                self.cell_line_response_df)
            pdx_weights = get_balanced_class_weights(self.pdx_response_df)

            cell_line_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                cell_line_weights, num_samples)
            pdx_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                pdx_weights, num_samples)

            # Data load with balanced sampling
            self._cell_line_dataloader = DataLoader(
                cell_line_dataset, batch_size=batch_size, sampler=cell_line_sampler)
            self._pdx_dataloader = DataLoader(
                pdx_dataset, batch_size=batch_size, sampler=pdx_sampler)

            # Paired sampling, matched batches
            if response_match_function is None:
                # def response_match_function(df_x, df_y):
                #    return torch.stack([df_x * x for x in df_y]).squeeze()

                def response_match_function(df_x, df_y):
                    return torch.stack([df_x * y for y in df_y]).squeeze() - torch.stack([torch.where(df_x != y, torch.ones_like(df_x), torch.zeros_like(df_x)) for y in df_y]).squeeze()

            for batch_id, ((cell_line_samples, cell_line_response), (pdx_samples, pdx_response)) in enumerate(zip(self._cell_line_dataloader, self._pdx_dataloader)):
                response_match = response_match_function(
                    cell_line_response, pdx_response)
                self.batches[batch_id] = (cell_line_samples,
                                          pdx_samples,
                                          response_match.to(self.device),
                                          cell_line_response.to(
                                              self.device),
                                          pdx_response.to(self.device))

        def create_dataset_full(self, response_match_function):
            # Data load with sequential sampling
            drop_last = cell_line_dataset.shape[0] % batch_size < 2
            self._cell_line_dataloader = DataLoader(
                cell_line_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
            self._pdx_dataloader = DataLoader(
                pdx_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

            # Sequential sampling, all batches
            batch_id = 0
            full_num_samples = 0

            if response_match_function is None:
                def response_match_function(df_x, df_y):
                    return torch.stack([df_x * x for x in df_y]).squeeze()

            for i, (cell_line_samples, cell_line_response) in enumerate(self._cell_line_dataloader):
                for j, (pdx_samples, pdx_response) in enumerate(self._pdx_dataloader):

                    response_match = response_match_function(
                        cell_line_response, pdx_response)
                    self.batches[batch_id] = (cell_line_samples,
                                              pdx_samples,
                                              response_match.to(self.device),
                                              cell_line_response.to(
                                                  self.device),
                                              pdx_response.to(self.device))
                    batch_id += 1
                    full_num_samples += batch_size

            self.num_samples = full_num_samples

        # create_dataset_full(self, response_match_function)
        create_dataset_balanced(self, response_match_function)
        if serialize:
            self._serialize()

    def get_cell_line_rna_dim(self):
        return self.cell_line_rna_df.shape[1]

    def get_pdx_rna_dim(self):
        return self.pdx_rna_df.shape[1]

    def __len__(self):
        return np.int64(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        return self.batches[index]


class ResponseDataloadersFactory:
    def __init__(self,
                 cell_line_source,
                 gene_set,
                 cross_validation_num,
                 validation_size=0.2,
                 random_seed=2023,
                 device=None):
        id_columns = ['Sample', 'UniqueID']

        self.drug_data = get_drug_data()
        self.cell_line_rna = get_cell_line_expression(gene_set=gene_set)
        self.cell_line_response = get_cell_line_response(cell_line_source)
        self.cell_line_response.columns = ['Sample', 'UniqueID', 'Response']
        self.binarized_cell_line_response = deepcopy(self.cell_line_response)
        self.binarized_cell_line_response['Response'] = binarize_auc_response(
            self.cell_line_response['Response'])
        self.cell_line_metadata = get_metadata()

        self.drug_data.set_index(id_columns[1], inplace=True)
        self.cell_line_response.set_index(id_columns[1], inplace=True)
        self.binarized_cell_line_response.set_index(
            id_columns[1], inplace=True)
        self.cell_line_rna.set_index(id_columns[0], inplace=True)
        self.cell_line_metadata.set_index(id_columns[0], inplace=True)

        self.nih_rna = get_nih_pdx_expression(gene_set=gene_set)
        self.nih_rna['Sample'] = [
            '-'.join(x.split('~')) for x in self.nih_rna['Sample']]
        self.nih_response = get_nih_pdx_response()
        #self.nih_response = expand_nih_response_dataset(
        #    self.nih_response, self.nih_rna['Sample'])
        #
        #self.nih_response = map_nci_drug_id(self.nih_response)
        #self.nih_response = self.nih_response[['Sample', 'Drug', 'Response']]
        self.nih_response.columns = ['Sample', 'UniqueID', 'Response']
        self.nih_metadata = get_metadata()
        self.nih_response.set_index('Sample', inplace=True)
        samples = np.intersect1d(self.nih_rna.index, self.nih_response.index)
        self.nih_response = self.nih_response.loc[samples]
        self.nih_response.reset_index(inplace=True)

        self.nih_response.set_index(id_columns[1], inplace=True)
        self.nih_rna.set_index(id_columns[0], inplace=True)
        self.nih_metadata.set_index(id_columns[0], inplace=True)

        self.nih_rna = self.nih_rna.loc[samples]
        self.nih_metadata = self.nih_metadata.loc[samples]
        
        self.cell_line_response_data_splits = {}
        self.pdx_response_data_splits = {}
        self.cross_validation_num = cross_validation_num
        self.validation_size = validation_size
        self.random_seed = random_seed

        self._drug_ids = np.unique(self.drug_data.index)
        self.device = device

        self.cell_line_sample_encoder = LabelEncoder()
        self.pdx_sample_encoder = LabelEncoder()
        self.cell_line_rna.index = self.cell_line_sample_encoder.fit_transform(
            self.cell_line_rna.index)
        self.nih_rna.index = self.pdx_sample_encoder.fit_transform(
            self.nih_rna.index)
        sample_column = 'Sample'
        self.cell_line_response[sample_column] = self.cell_line_sample_encoder.transform(
            self.cell_line_response[sample_column])
        self.binarized_cell_line_response[sample_column] = self.cell_line_sample_encoder.transform(
            self.binarized_cell_line_response[sample_column])
        self.nih_response[sample_column] = self.pdx_sample_encoder.transform(
            self.nih_response[sample_column])

        self._prepare_drugwise_response_cv_splits(
            self.cross_validation_num, self.validation_size, self.random_seed)

    def _prepare_drugwise_response_cv_splits(self, cross_validation_num, validation_size=0.2, random_seed=2023):

        def create_train_val_test_splits(df, strat_condition, cross_validation_num, validation_size, random_seed):
            df.reset_index(inplace=True)
            cv_splits = {}
            test_size = 1./cross_validation_num
            cell_line_splitter = StratifiedShuffleSplit(
                n_splits=cross_validation_num, test_size=test_size, random_state=random_seed)
            cell_line_cv = cell_line_splitter.split(
                strat_condition, strat_condition)

            for cv_idx, (cl_global_train_idx, cl_test_idx) in enumerate(cell_line_cv):
                cv_splits[cv_idx] = {}
                val_splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=validation_size, random_state=random_seed)
                cl_train_idx, cl_val_idx = next(val_splitter.split(strat_condition.iloc[cl_global_train_idx],
                                                                   strat_condition.iloc[cl_global_train_idx]))
                # train_test_split(strat_condition[cl_global_train_idx],
                #                 test_size=validation_size,
                #                 random_state=random_seed,
                #                 stratify=strat_condition[cl_global_train_idx])
                cl_train_idx = cl_global_train_idx[cl_train_idx]
                cl_val_idx = cl_global_train_idx[cl_val_idx]
                df['UniqueID'] = [str(x) for x in df['UniqueID'].values]
                #if type(df['UniqueID'].loc[0]) is str:
                df['UniqueID'] = [int(x.split('_')[-1]) for x in df['UniqueID'].values]
                df['Response'] = df['Response'].astype(float)
                #breakpoint()
                cv_splits[cv_idx]['train'] = torch.tensor(df.loc[df.index[cl_train_idx]].values).to(self.device)
                cv_splits[cv_idx]['val'] = torch.tensor(df.loc[df.index[cl_val_idx]].values).to(self.device)
                cv_splits[cv_idx]['test'] = torch.tensor(df.loc[df.index[cl_test_idx]].values).to(self.device)

            return cv_splits

        #breakpoint()
        for drug_id in np.unique(self.drug_data.index):

            if drug_id in self.cell_line_response.index:
                drug_cell_line_response = self.cell_line_response.loc[drug_id]
                binarized_drug_cell_line_response = self.binarized_cell_line_response.loc[
                    drug_id]
                if sum(binarized_drug_cell_line_response['Response']) < 2:
                    continue
            
                self.cell_line_response_data_splits[drug_id] = create_train_val_test_splits(drug_cell_line_response,  # binarized_drug_cell_line_response, #drug_cell_line_response,
                                                                                            binarized_drug_cell_line_response[
                                                                                                'Response'],
                                                                                            cross_validation_num=cross_validation_num,
                                                                                            validation_size=validation_size,
                                                                                            random_seed=random_seed)

            selected_pdo_drugs =['Drug_1036', 'Drug_1103', 'Drug_1418', 'Drug_1493', 'Drug_293', 'Drug_384', 'Drug_384']
            if drug_id in self.nih_response.index:
                drug_nih_response = self.nih_response.loc[drug_id]
                #if drug_nih_response.shape[0] < 10:
                #    continue
                #breakpoint()
                if drug_id not in selected_pdo_drugs:
                    continue
                if sum(drug_nih_response['Response']) < 2 or drug_nih_response.shape[0]-sum(drug_nih_response['Response']) < 2:
                    #if sum(drug_nih_response['Response']) > 0:
                    #    print(drug_id)
                    continue

                self.pdx_response_data_splits[drug_id] = create_train_val_test_splits(drug_nih_response,
                                                                                      drug_nih_response['Response'],
                                                                                      cross_validation_num=cross_validation_num,
                                                                                      validation_size=validation_size,
                                                                                      random_seed=random_seed)

    def _get_dataloaders(self, split, rna_df, batch_size, num_samples=None, device=None):
        dataloaders = {}
        for key in split:

            rna_df = rna_df
            dataset = ResponseDataset(response_df=split[key],
                                      rna_df=rna_df,
                                      device=device)
            drop_last = len(dataset) % batch_size < 2
            if key == 'train':
                if num_samples is not None:
                    # weights = get_stratified_class_weights(split[key])
                    # weights = get_balanced_class_weights(split[key])
                    weights = torch.ones(split[key].shape[0])
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(
                        weights, num_samples=num_samples)
                    dataloaders[key] = DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=sampler,
                                                  drop_last=drop_last)
                else:
                    dataloaders[key] = DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=drop_last)
            else:
                dataloaders[key] = DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=drop_last)
            # collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x))
        return dataloaders['train'], dataloaders['val'], dataloaders['test']

    def get_drug_specific_cell_line_dataloaders(self, drug_id, cv_idx, batch_size=128, num_samples=None):
        response_split = self.cell_line_response_data_splits[drug_id][cv_idx]
        unique_samples = torch.cat(
            [response_split[key][:,1] for key in response_split.keys()]) # 'Sample' column stands for 1
        rna_df = self.cell_line_rna.loc[unique_samples.detach().int().to('cpu')]
        rna_df = rna_df[~rna_df.index.duplicated(keep='first')]
        return self._get_dataloaders(split=response_split,
                                     rna_df=rna_df,
                                     batch_size=batch_size,
                                     num_samples=num_samples,
                                     device=self.device)

    def get_drug_specific_pdx_dataloaders(self, drug_id, cv_idx, batch_size=128, num_samples=None):
        response_split = self.pdx_response_data_splits[drug_id][cv_idx]
        unique_samples = torch.cat(
            [response_split[key][:,1] for key in response_split.keys()]) # 'Sample' column stands for :,1
        rna_df = self.nih_rna.loc[unique_samples.detach().int().to('cpu')]
        rna_df = rna_df[~rna_df.index.duplicated(keep='first')]
        return self._get_dataloaders(split=response_split,
                                     rna_df=rna_df,
                                     batch_size=batch_size,
                                     num_samples=num_samples,
                                     device=self.device)

    def get_paired_dataloaders_keys(self):
        keys = {}
        for drug_id in self._drug_ids:

            if drug_id not in self.cell_line_response_data_splits or drug_id not in self.pdx_response_data_splits:
                continue

            keys[drug_id] = {}

            for cv_idx in range(self.cross_validation_num):
                keys[drug_id][cv_idx] = {}
        return keys

    def get_paired_cell_line_pdx_loaders(self, drug_id, cv_idx, num_samples, batch_size=128):
        drug_dataloaders = {}
        cell_line_response_split = deepcopy(
            self.cell_line_response_data_splits[drug_id][cv_idx])
        pdx_response_split = self.pdx_response_data_splits[drug_id][cv_idx]
        for set_name in cell_line_response_split:
            unique_cell_line_samples = torch.unique(
                cell_line_response_split[set_name][:,1]) # sample columns ['Sample']
            unique_pdx_samples = torch.unique(
                pdx_response_split[set_name][:,1]) # sample columns ['Sample']
            cell_line_response_split[set_name][:,2] = binarize_auc_response(
                cell_line_response_split[set_name][:,2]) # response columns ['Response']

            effective_num_samples = num_samples
            if set_name != 'train':
                effective_num_samples = 1000

            paired_dataset = DrugSpecificPairedDataset(effective_num_samples,
                                                       batch_size,
                                                       self.cell_line_rna.loc[unique_cell_line_samples.to('cpu').detach()],
                                                       cell_line_response_split[set_name],
                                                       self.nih_rna.loc[unique_pdx_samples.to('cpu').detach()],
                                                       pdx_response_split[set_name],
                                                       cell_line_indices=self.cell_line_rna.index,
                                                       pdx_indices=self.nih_rna.index,
                                                       cell_line_rna_metadata_df=None,
                                                       pdx_rna_metadata_df=None,
                                                       response_match_function=None,
                                                       device=self.device)
            drug_dataloaders[set_name] = DataLoader(
                paired_dataset, batch_size=1)
        return drug_dataloaders['train'], drug_dataloaders['val'], drug_dataloaders['test']


def get_data_generator(dataloader: DataLoader):
    if dataloader is None:
        return None
    return dataloader.__iter__()


def get_kegg(organism='hsa'):
    if os.path.isfile(KEGG_FILE):
        return pickle.load(open(KEGG_FILE, 'rb'))
    ref_pathways = np.array(kegg_list('path').read().split('\n'))
    # print(ref_pathways)

    processed_pathways = []
    for pathway in ref_pathways:
        if len(pathway) > 0:
            try:
                processed_pathways.append(
                    pathway.split('\t')[0].split('map')[1])
            except:
                continue
    ref_pathways = processed_pathways

    # hsa_entries = np.array(kegg_list('hsa').read().split('\n'))
    # hsa_entries = np.array([x.split('\t')[0].split(':')[1] for x in hsa_entries if len(x) > 0])
    # hsa_entries = np.array([x for x in hsa_entries if (len(x) == 5)])

    # print(np.shape(ref_pathways))
    # print(ref_pathways)

    org_pathways = []
    for path_code in ref_pathways:
        # print(path_code)
        pathway = None
        try:
            pathway = kegg_get('{}{}'.format(organism, path_code)).read()
        except:
            try:
                pathway = kegg_get('{}{}'.format('map', path_code)).read()
            except:
                continue
        org_pathways.append(pathway)
    # print(np.array(org_pathways))
    # print(np.shape(org_pathways))
    pickle.dump(org_pathways, open(KEGG_FILE, 'wb'))
    return org_pathways


def get_genes(pathways):
    processed_pathways = {}
    pathway_names = {}
    keggId2symbol = {}
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
                gene = tokens[-1]
                kegg_id = tokens[-3]
                keggId2symbol[kegg_id] = gene
                processed_pathways[pathway_name].append(gene)

    return processed_pathways, pathway_names, keggId2symbol


def get_pathways():
    kegg = get_kegg()
    return get_genes(kegg)


def fetch_kegg_pathway_hierarchy():
    # Fetch the list of all pathways
    # pathway_list = kegg_list("pathway").read()
    pathway_list, pathway_codes = get_pathways()

    # Dictionary to store the hierarchy
    pathway_hierarchy = {}

    # Process each line in the pathway list
    for pathway_name, pathway_id in pathway_codes.items():
        # Extract pathway category from the pathway name
        category = pathway_name.split(" - ")[0]

        if category not in pathway_hierarchy:
            pathway_hierarchy[category] = []

        # Add pathway to the respective category
        pathway_hierarchy[category].append((pathway_id, pathway_name))

    return pathway_hierarchy


if __name__ == "__main__":
    cross_validation_num = 5
    dataloader_factory = ResponseDataloadersFactory(
        cell_line_source=Source.CCLE,
        gene_set=GeneSet.ALL,
        cross_validation_num=cross_validation_num,
        validation_size=0.2,
        random_seed=2023)

    keys = dataloader_factory.get_paired_dataloaders_keys()
    for drug_id in keys:
        for cv_split in keys[drug_id]:
            paired_train_dataloaders, paired_val_dataloader, paired_test_dataloader = \
                dataloader_factory.get_paired_cell_line_pdx_loaders(
                    drug_id, cv_split, num_samples=1000, batch_size=64)
            cell_line_train_loader, cell_line_val_loader, cell_line_test_loader = \
                dataloader_factory.get_drug_specific_cell_line_dataloaders(
                    drug_id, cv_split, num_samples=1000, batch_size=256)
            pdx_train_loader, pdx_val_loader, pdx_test_loader \
                = dataloader_factory.get_drug_specific_pdx_dataloaders(drug_id, cv_split, num_samples=1000, batch_size=256)

            # for i, (cell_rna, auc) in enumerate(cell_line_train_loader):
            #    breakpoint()
            #    pass
            # for i, (cell_rna, pdx_rna, matches) in enumerate(paired_train_dataloaders):
            #    breakpoint()
            #    pass
