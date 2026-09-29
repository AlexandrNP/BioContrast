import os
import math
import torch
import itertools
import numpy as np
from copy import deepcopy
from torch import nn, vmap
# from functorch import vmap
from typing import overload
from torch.nn import functional as F
from abc import ABC, ABCMeta, abstractmethod
from kegg import construct_kegg_hierarchies
#import sparselinear as sl  # disabled: torch_sparse not installed (matches PDO folders)

# Convolution-kernel init is selectable via env var so filter variants can be swept:
#   BC_CONV_INIT in {default, kaiming, kaiming_uniform, xavier, orthogonal}
_CONV_INIT = os.environ.get('BC_CONV_INIT', 'default')

# --------------------------------------------------------------------------- #
# B2: Mechanism-of-Action (MoA) pathway gating (OPT-IN, default OFF).          #
#                                                                             #
# When BC_MOA_GATE=1, the gene->pathway heads of the KEGG-CNN (the first       #
# MultiHeadLayer, whose heads ARE the KEGG pathways) are multiplicatively      #
# gated by a per-drug target-pathway row T[drug] over the pathway names:       #
#     z_gated = z_head * (1 + BC_MOA_GAMMA * T[drug, pathway_of_head])         #
# So a drug's *relevant* pathway heads are up-weighted (T=1 -> x(1+gamma)) and #
# irrelevant heads pass through unchanged (T=0 -> x1). This lets the model     #
# read a targeted drug's response from expression shifts in the pathways that  #
# drug actually hits -- WITHOUT any mutation data -- and is fully reversible:  #
# with the flag off the forward is byte-for-byte the original.                 #
# The drug is selected per-process by BC_MOA_DRUG (a Drug_<id>), or set        #
# programmatically via CellLineTransferLearner.set_moa_target(drug_id).        #
_MOA_GATE = os.environ.get('BC_MOA_GATE', '0') == '1'
_MOA_GAMMA = float(os.environ.get('BC_MOA_GAMMA', '1.0'))


# Drug (de-anonymised MoA) -> the KEGG pathway names it mechanistically hits.
# Names are matched case-insensitively against the model's actual pathway heads;
# any name not present as a head is simply skipped (gate stays neutral there).
_MOA_DRUG_PATHWAYS = {
    # EGFR / HER2 tyrosine-kinase inhibitors
    'Drug_1312': ['ErbB signaling pathway', 'MAPK signaling pathway',            # Osimertinib
                  'EGFR tyrosine kinase inhibitor resistance', 'PI3K-Akt signaling pathway'],
    'Drug_883':  ['ErbB signaling pathway', 'MAPK signaling pathway',            # Erlotinib
                  'EGFR tyrosine kinase inhibitor resistance', 'PI3K-Akt signaling pathway'],
    'Drug_1405': ['ErbB signaling pathway', 'MAPK signaling pathway',            # Gefitinib
                  'EGFR tyrosine kinase inhibitor resistance', 'PI3K-Akt signaling pathway'],
    'Drug_520':  ['ErbB signaling pathway', 'MAPK signaling pathway',            # Afatinib (pan-HER)
                  'EGFR tyrosine kinase inhibitor resistance'],
    'Drug_184':  ['ErbB signaling pathway', 'MAPK signaling pathway',            # Neratinib (pan-HER)
                  'EGFR tyrosine kinase inhibitor resistance'],
    'Drug_1391': ['ErbB signaling pathway', 'MAPK signaling pathway',            # Sapitinib
                  'EGFR tyrosine kinase inhibitor resistance'],
    # MEK inhibitor
    'Drug_384':  ['MAPK signaling pathway', 'Ras signaling pathway'],            # Trametinib (MEKi)
    # Other RTK / non-receptor TK inhibitors
    'Drug_364':  ['MAPK signaling pathway', 'Focal adhesion',                    # Dasatinib (SRC/ABL)
                  'PI3K-Akt signaling pathway'],
    'Drug_715':  ['B cell receptor signaling pathway', 'NF-kappa B signaling pathway',  # Ibrutinib (BTK)
                  'MAPK signaling pathway'],
    'Drug_772':  ['JAK-STAT signaling pathway', 'MAPK signaling pathway'],       # Lestaurtinib (FLT3/JAK2)
    'Drug_538':  ['Ras signaling pathway', 'MAPK signaling pathway'],            # Tipifarnib (FTase/RAS)
    'Drug_614':  ['PI3K-Akt signaling pathway',                                  # HB3599 (HSP90 ansamycin)
                  'Protein processing in endoplasmic reticulum'],
    # BCL2 / apoptosis
    'Drug_1103': ['Apoptosis', 'p53 signaling pathway'],                         # Navitoclax (BCL2/xL)
    'Drug_1327': ['Apoptosis', 'p53 signaling pathway'],                         # Obatoclax (BCL2)
    'Drug_1078': ['Apoptosis', 'p53 signaling pathway'],                         # Sepantronium/YM155 (survivin)
    # Cell-cycle / mitotic (taxanes, vinca, PLK1, WEE1, topo)
    'Drug_1105': ['Cell cycle'],                                                 # Docetaxel (taxane)
    'Drug_1127': ['Cell cycle'],                                                 # Paclitaxel (taxane)
    'Drug_1418': ['Cell cycle'],                                                 # Vinblastine (vinca)
    'Drug_1472': ['Cell cycle'],                                                 # BI-2536 (PLK1)
    'Drug_554':  ['Cell cycle', 'MAPK signaling pathway'],                       # Tirbanibulin (SRC/tubulin)
    'Drug_572':  ['Cell cycle', 'p53 signaling pathway'],                        # Adavosertib/MK-1775 (WEE1)
    'Drug_1493': ['Cell cycle', 'p53 signaling pathway', 'DNA replication'],     # Doxorubicin (topo II)
    'Drug_194':  ['Cell cycle', 'DNA replication'],                              # SN-38 (topo I)
    # Proteasome / neddylation
    'Drug_988':  ['Proteasome', 'Ubiquitin mediated proteolysis'],              # Bortezomib
    'Drug_908':  ['Proteasome', 'Ubiquitin mediated proteolysis'],              # Pevonedistat (NAE)
    # Metabolic / antimetabolite
    'Drug_1036': ['One carbon pool by folate', 'Folate biosynthesis',           # Methotrexate (DHFR)
                  'Purine metabolism', 'Pyrimidine metabolism'],
    'Drug_865':  ['Pyrimidine metabolism', 'DNA replication'],                   # Gemcitabine
    'Drug_293':  ['Nicotinate and nicotinamide metabolism'],                     # Daporinad (NAMPT)
    'Drug_984':  ['Oxidative phosphorylation'],                                  # Oligomycin A (ATP synthase)
}


def moa_target_pathways(drug_id):
    """Return the list of KEGG pathway names drug_id mechanistically targets (or [])."""
    return _MOA_DRUG_PATHWAYS.get(drug_id, [])


def _init_conv_weight(layer):
    """Initialize a per-head conv-kernel weight according to BC_CONV_INIT."""
    if not hasattr(layer, 'weight'):
        return
    w = layer.weight
    if _CONV_INIT == 'kaiming':
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    elif _CONV_INIT == 'kaiming_uniform':
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    elif _CONV_INIT == 'xavier':
        nn.init.xavier_normal_(w)
    elif _CONV_INIT == 'orthogonal':
        nn.init.orthogonal_(w)
    # 'default': leave nn.Linear's own initialization untouched


def get_linear_layer(in_features, out_features, sparse=False, bias=True):
    # sparselinear disabled (torch_sparse unavailable); always dense, matching PDO folders.
    return nn.Linear(in_features, out_features, bias=bias)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def initialize_non_glu(module, input_dim, output_dim):
    # Adopted from TabNet
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, input_dim, output_dim):
    # Adopted from TabNet
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    # Adopted from TabNet
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=256, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        # Small-data guard: a size-1 eval/test fold can arrive as a 1D [features] tensor after upstream
        # squeezes; BatchNorm1d rejects 1D. Promote to [1, features] (a batch of 1) -- valid in eval mode
        # (running stats). Training size-1 batches are skipped upstream, so this only bites at inference.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class FilterBase(nn.Module, ABC):
    def __init__(self):
        super(FilterBase, self).__init__()
        self.in_features = -1
        self.out_features = -1
        self._preprocessing_modules = [nn.Identity()]
        self._preprocessing_layers = None
        self._is_instantiated = False

    @abstractmethod
    def initialize(self, in_features):
        raise NotImplementedError

    @abstractmethod
    def get_out_features_num(self):
        raise NotImplementedError

    def add_preprocessing(self, module: nn.Module):
        self.preprocessing_modules.append(module)

    def _preprocess(self, x):
        if self._preprocessing_layers is None:
            self._preprocessing_layers = nn.Sequential(
                *self._preprocessing_modules)
        return self._preprocessing_layers(x)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class Conv1dFilter(FilterBase):
    def __init__(self, kernel_size=3, padding=0, bias=False, device='cpu', sparse=False):
        super(Conv1dFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.device = device
        self.sparse = sparse

    def initialize(self, in_features):
        self.in_features = in_features
        self.out_features = self.in_features + \
            2 * self.padding - (self.kernel_size-1)
        self.weights = get_linear_layer(
            in_features=self.kernel_size, out_features=1, bias=self.bias, sparse=self.sparse)
        # Per-head conv-kernel init, selectable via BC_CONV_INIT for the filter sweep.
        _init_conv_weight(self.weights)
        self.batched_kernel = vmap(
            lambda batch, weights=self.weights: weights(batch))
        self.initialized = True

    def get_out_features_num(self):
        return self.out_features

    def forward(self, x):
        if not self.initialized:
            raise Exception(
                f"An instance of the class {type(self)} has to be initialized before usage! This class has a lazy initialization")
        if self.padding > 0:
            #breakpoint()
            x = torch.concat((torch.zeros(x.shape[0], self.padding).to(device=self.device), x, torch.zeros(x.shape[0], self.padding).to(device=self.device)), dim=-1)
        # B1: x is [B, in]; a bare .squeeze() collapsed a size-1 batch to 1D and broke
        # unfold(1,...). x is already 2D here, so preprocess it directly (Identity),
        # and squeeze ONLY the trailing kernel-output axis after the per-head kernel.
        preprocessed_x = self._preprocess(x)
        preprocessed_x = preprocessed_x.unfold(1, self.kernel_size, 1)
        return self.batched_kernel(preprocessed_x).squeeze(-1)


class LinearProjectionFilter(FilterBase):
    """Dense per-head projection (global mixing) instead of a local conv kernel.
    Same output size as Conv1dFilter so the hierarchy dims are identical (fair operator test)."""
    def __init__(self, kernel_size=3, padding=0, bias=True, device='cpu', sparse=False, **kw):
        super(LinearProjectionFilter, self).__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.device = device

    def initialize(self, in_features):
        self.in_features = in_features
        self.out_features = max(1, in_features - (self.kernel_size - 1))
        self.proj = nn.Linear(in_features, self.out_features, bias=self.bias)
        _init_conv_weight(self.proj)
        self.initialized = True

    def get_out_features_num(self):
        return self.out_features

    def forward(self, x):
        return self.proj(x.squeeze())


class _PoolFilter(FilterBase):
    """Fixed (non-learned) windowed pooling kernel — mean or max — same out size as Conv1d."""
    MODE = 'mean'
    def __init__(self, kernel_size=3, padding=0, bias=False, device='cpu', sparse=False, **kw):
        super(_PoolFilter, self).__init__()
        self.kernel_size = kernel_size; self.device = device

    def initialize(self, in_features):
        self.in_features = in_features
        self.out_features = max(1, in_features - (self.kernel_size - 1))
        self.initialized = True

    def get_out_features_num(self):
        return self.out_features

    def forward(self, x):
        x = x.squeeze()
        if x.dim() < 2 or x.shape[-1] < self.kernel_size:
            return x
        w = x.unfold(1, self.kernel_size, 1)          # (batch, out, kernel)
        return w.mean(-1) if self.MODE == 'mean' else w.max(-1).values


class MeanPoolFilter(_PoolFilter):
    MODE = 'mean'


class MaxPoolFilter(_PoolFilter):
    MODE = 'max'


_KEGG_ADJ_CACHE = {}


def _build_kegg_adjacency(keggid2symbol):
    """Gene-symbol -> set(neighbour symbols) from KEGG per-pathway edge lists
    (Entrez ids in KEGG/KEGGID/hsa*.csv mapped to symbols). Cached module-wide."""
    if 'adj' in _KEGG_ADJ_CACHE:
        return _KEGG_ADJ_CACHE['adj']
    import glob
    import pandas as pd
    from data import KEGGID_DIR
    adj = {}
    for f in glob.glob(os.path.join(KEGGID_DIR, 'hsa*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue  # some pathway files are empty
        if 'from' not in d.columns or 'to' not in d.columns:
            continue
        for a, b in zip(d['from'], d['to']):
            sa = keggid2symbol.get(str(a))
            sb = keggid2symbol.get(str(b))
            if sa is None or sb is None or sa == sb:
                continue
            adj.setdefault(sa, set()).add(sb)
            adj.setdefault(sb, set()).add(sa)
    _KEGG_ADJ_CACHE['adj'] = adj
    return adj


class GraphConvFilter(FilterBase):
    """Graph convolution over a pathway's real KEGG gene-gene adjacency.

    Where Conv1dFilter slides a kernel over genes in an arbitrary (alphabetical)
    order, this mixes each gene with its *actual* KEGG pathway neighbours. To
    avoid the classic GNN over-smoothing collapse (a naive 2-hop A_hat@A_hat on
    ~13%-dense pathway graphs averages every gene toward the pathway mean, and the
    downstream per-head BatchNorm then collapsed the contrastive embedding to a
    constant in eval mode -> AUC 0.5), it uses a single 1-hop propagation with
    GraphSAGE-style self||neighbour concat and an output LayerNorm whose scale is
    identical in train and eval (no BatchNorm train/eval drift):
        z = A_hat @ x                         (1-hop neighbour aggregate)
        h = GELU(W1 [x || z])                 (self + neighbour, per gene)
        y = LayerNorm(W2 h)                   (one stable feature per gene)
    `adj` is a dense (n,n) 0/1 tensor for the head's genes in input-column order.
    Above the gene level (no gene-gene graph) adj is None -> dense linear mix."""
    HIDDEN = 16

    def __init__(self, kernel_size=3, padding=0, bias=True, device='cpu', sparse=False, adj=None, **kw):
        super(GraphConvFilter, self).__init__()
        self.device = device
        self.bias = bias
        self._adj_raw = adj
        self._mode = None

    def _norm_adj(self, n):
        a = self._adj_raw.float()
        a = a + torch.eye(n)                      # self-loops
        d = a.sum(1).clamp(min=1e-6)
        dinv = d.pow(-0.5)
        return dinv.unsqueeze(1) * a * dinv.unsqueeze(0)   # sym-normalised A_hat

    def initialize(self, in_features):
        self.in_features = in_features
        self.out_features = in_features           # one output feature per gene node
        H = self.HIDDEN
        if self._adj_raw is not None and int(in_features) == int(self._adj_raw.shape[0]):
            self.register_buffer('ahat', self._norm_adj(in_features).to(self.device))
            self.lin1 = nn.Linear(2, H, bias=self.bias)      # [self || neighbour]
            self.lin2 = nn.Linear(H, 1, bias=self.bias)      # readout
            self.onorm = nn.LayerNorm(int(in_features))      # train/eval-stable scale
            self.act = nn.GELU()
            self._mode = 'graph'
        else:
            # no KEGG graph at this level -> dense per-head mixing
            self.proj = nn.Linear(in_features, in_features, bias=self.bias)
            _init_conv_weight(self.proj)
            self._mode = 'dense'
        self.initialized = True

    def get_out_features_num(self):
        return self.out_features

    def forward(self, x):
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self._mode == 'dense':
            return self.proj(x)
        xf = x.unsqueeze(-1)                                   # (b, n, 1)
        z = torch.einsum('ij,bjc->bic', self.ahat, xf)        # (b, n, 1) 1-hop
        h = self.act(self.lin1(torch.cat([xf, z], dim=-1)))   # (b, n, H) self||nbr
        y = self.lin2(h).squeeze(-1)                          # (b, n)
        return self.onorm(y)                                  # (b, n) stable scale


class FilterBundle(nn.Module):
    def __init__(self):
        super(FilterBundle, self).__init__()
        self.filters = nn.ModuleList([])
        # self.filter_masks = []
        self.input_features_num = None
        self.output_maps = None
        self.initialized = False
        self.device = 'cpu'
        # self.previous_filter_maps = None
        # self.filter_params = []

    def set_device(self, device):
        self.device = device

    def add_filter(self, filter: FilterBase, filter_mask=None):
        self.filters.append(filter)
        # self.filter_masks.append(filter_mask)

    def set_input_features_num(self, input_features_num: int):
        self.input_features_num = input_features_num

    def _initialize_output_maps(self):
        if self.output_maps is None:
            self.output_maps = []
            for i in range(len(self.filters)):
                self.output_maps = self.output_maps + \
                    list(itertools.repeat(
                        i, self.filters[i].get_out_features_num()))

        self.output_maps = torch.tensor(self.output_maps).to(self.device)

    def _initialize_filters(self, in_features_num):
        for i in range(len(self.filters)):
            self.filters[i].initialize(in_features_num)

    def _initialize(self):
        if self.input_features_num is None:
            raise Exception(
                f"Cannot initialize {type(self)}. Unknown input features number")
        self._initialize_filters(self.input_features_num)
        self._initialize_output_maps()
        self.initialized = True

    def get_out_channels(self):
        return self.output_maps.squeeze()

    def forward(self, x):
        if not self.initialized:
            self._initilize()

        outputs = []
        for conv_filter in self.filters:
            filter_output = conv_filter(x)
            # keep a (batch, out_i) shape so a multi-filter combination concatenates
            # along features, not batch (a lone filter that squeezed to 1-D is restored)
            if filter_output.dim() == 1:
                filter_output = filter_output.unsqueeze(-1)
            outputs.append(filter_output)

        # Concatenate every filter's channels -> (batch, sum_i out_i). dim=-1 (not dim=0)
        # so filter COMBINATIONS stack on the feature axis; harmless for a single filter.
        # B1: NO trailing .squeeze() -- each output is already [B, out_i] so the cat is
        # 2D [B, F]; squeezing collapsed the batch dim when B==1.
        return torch.cat(outputs, dim=-1)


class FilterBundleFactory():
    def __init__(self):
        super(FilterBundleFactory, self).__init__()
        self.filters = []
        self.filter_params = []

    def register_filter(self, filter):
        self.filters.append(filter)

    # def init_filters(self):
    #    self.kernel_list = nn.ModuleList([])
    #    for col in self.hierarchy_mapping.T:
    #        col.nonzero(as_tuple=True)[0]

    def get_filter_bundle(self, input_features_num):
        filter_bundle = FilterBundle()
        for conv_filter in self.filters:
            filter_bundle.add_filter(conv_filter)
        filter_bundle.set_input_features_num(input_features_num)
        filter_bundle._initialize()

        return filter_bundle


class HierarchyUpdater():
    # Knows the number of elements in the previous hierarchy
    # Expands mapping of the hierarchy level to fit the updated number of channels
    # Should be updated after each convolutional layer application
    def __init__(self, channels_per_hierarchy, device):
        # self.channels_per_hierarchy = channels_per_hierarchy
        self.device = device
        self.update(channels_per_hierarchy)

    def update(self, channels_per_hierarchy):
        # torch.tensor(channels_per_hierarchy).T
        self.channels_per_hierarchy = channels_per_hierarchy
        self.remaining_hierarchies = torch.cat(
            self.channels_per_hierarchy).unique()

    def __call__(self, hierarchy_map):
        hierarchy_map = hierarchy_map.to(self.device)[self.remaining_hierarchies]
        expanded = [torch.kron(channel_idx, hierarchy_map[i])
                    for i, channel_idx in enumerate(self.channels_per_hierarchy)]
        return torch.cat(expanded, dim=-1).squeeze()


class HierarchyMap(nn.Module):
    # Expected input hierarchy mapping is a 2D boolean matrix with the
    # dimensions in_features x out_hierarchical_objects
    def __init__(self, device: str, hierarchy_mapping: torch.Tensor, input_channel_mapping: torch.Tensor, hierarchy_updater: HierarchyUpdater):
        super(HierarchyMap, self).__init__()
        # self.stub_layers = nn.ModuleList([])
        self.device = device
        self.hierarchy_mapping_channels = hierarchy_mapping
        self.hierarchy_mapping_idx = []
        self.channel_mapping = []
        for i in range(self.hierarchy_mapping_channels.shape[1]):
            map_idx = None
            if hierarchy_updater is not None:
                #breakpoint()
                map_idx = hierarchy_updater(
                    self.hierarchy_mapping_channels[:, i]).nonzero()
            else:
                map_idx = self.hierarchy_mapping_channels[:, i].nonzero()
            self.hierarchy_mapping_idx.append(map_idx)
            self.channel_mapping.append(
                    torch.tensor([i]).repeat(map_idx.size()).to(self.device))
            #except:
            #    breakpoint()
        self.hierarchy_mapping_idx = torch.cat(self.hierarchy_mapping_idx)
        self.channel_mapping = torch.cat(self.channel_mapping)

    def get_channel_mapping(self):
        return self.channel_mapping.squeeze()

    def forward(self, x):
        # print(f'X shape: {x.shape}')
        # print(f'Mapping_max {self.hierarchy_mapping_idx.max()}')
        # print(f'Mapping_min {self.hierarchy_mapping_idx.min()}')
        # breakpoint()
        # B1: hierarchy_mapping_idx is [K,1] so x[:,idx] is [B,K,1]; squeeze ONLY the
        # trailing feature axis (squeeze(-1)) so a size-1 batch stays [1,K] instead of
        # collapsing to [K] (which broke the next layer's x[:,head] -> "too many indices").
        return x[:, self.hierarchy_mapping_idx].squeeze(-1)


class MultiHeadLayer(nn.Module):
    def __init__(self, custom_filter_classes, filter_head_size_param_names, input_head_mapping, device='cpu', filters_other_params=None, reduce_to_size=None, graph_gene_symbols=None, graph_adjacency=None, graph_col2gene=None):
        super(MultiHeadLayer, self).__init__()

        self.device = device
        self.input_head_mapping = input_head_mapping
        self._graph_gene_symbols = graph_gene_symbols
        self._graph_adjacency = graph_adjacency
        self._graph_col2gene = graph_col2gene
        self.output_channel_mapping = None
        # B2: per-head MoA gate multiplier (aligned to heads_out_idx order). None => no
        # gating (default). Set by KEGGHierarchicalCNN.apply_moa_target on the pathway layer.
        self._moa_gate = None
        # self.head_filter_bundle_factories = []
        self.head_filter_bundles = torch.nn.ModuleList([])
        self.head_inputs = []
        self.heads_out_channels_num = []
        self.heads_out_channels = []
        self.heads_out_idx = []

        filter_param_list = [dict() for _ in range(len(custom_filter_classes))]
        if filter_head_size_param_names is None:

            filter_head_size_param_names = [[]
                                            for _ in range(len(custom_filter_classes))]
        if filters_other_params is not None:
            assert len(custom_filter_classes) == len(filters_other_params)
            filter_param_list = filters_other_params

        head_list = torch.sort(torch.unique(self.input_head_mapping))[0]
        # breakpoint()
        channel_counter = 0
        self.norm_layers = torch.nn.ModuleList([])
        self.reduction_layers = torch.nn.ModuleList([])
        for head_idx in head_list:
            head_input = torch.where(
                self.input_head_mapping == head_idx, 1, 0).bool().squeeze()
            head_input_size = torch.sum(head_input)
            self.head_inputs.append(head_input)

            # KEGG graph conv: build this head's gene-gene adjacency submatrix.
            head_adj = None
            if self._graph_adjacency is not None and self._graph_col2gene is not None:
                # build the adjacency on CPU (indices/maps may live on different devices)
                exp_cols = torch.where(head_input.cpu())[0]
                gidx = self._graph_col2gene.cpu().reshape(-1)[exp_cols].tolist()
                syms = [self._graph_gene_symbols[g] for g in gidx]
                m = len(syms)
                head_adj = torch.zeros(m, m)
                pos = {}
                for a, s in enumerate(syms):
                    pos.setdefault(s, a)  # genes are unique within a pathway head
                for a, s in enumerate(syms):
                    for nb in self._graph_adjacency.get(s, ()):
                        j = pos.get(nb)
                        if j is not None:
                            head_adj[a, j] = 1.0
                            head_adj[j, a] = 1.0

            self.filter_bundle_factory = FilterBundleFactory()

            for conv_filter_class, filter_params, head_size_names in zip(custom_filter_classes, filter_param_list, filter_head_size_param_names):
                for name in head_size_names:
                    filter_params[name] = head_input_size
                current_filter_params = deepcopy(filter_params)
                current_filter_params['device'] = self.device
                if conv_filter_class is GraphConvFilter:
                    current_filter_params['adj'] = head_adj
                if 'kernel_size' in current_filter_params:
                    if current_filter_params['kernel_size'] == -1:
                        # Kernel size selectable via BC_CONV_KERNEL: half (default), third,
                        # full (global), or a fixed int (1,3,5,7...) capped to the head size.
                        _ks = os.environ.get('BC_CONV_KERNEL', 'half')
                        # cap so out_features = in-(k-1) stays >= 2 (a size-1 output squeezes
                        # to a 0-d tensor and breaks the hierarchy's len()).
                        _cap = max(1, head_input_size - 1)
                        if _ks == 'half':
                            kernel_size = max(1, head_input_size // 2)
                        elif _ks == 'third':
                            kernel_size = max(1, head_input_size // 3)
                        elif _ks == 'full':
                            kernel_size = _cap
                        else:
                            kernel_size = max(1, int(_ks))
                        kernel_size = min(kernel_size, _cap)
                        current_filter_params['kernel_size'] = kernel_size
                        current_filter_params['padding'] = 0
                conv_filter = conv_filter_class(**current_filter_params)
                self.filter_bundle_factory.register_filter(conv_filter)
            head_filter_bundle = self.filter_bundle_factory.get_filter_bundle(
                head_input_size)
            head_filter_bundle.set_device(self.device)
            new_head_features_num = len(  # new_channels_num
                head_filter_bundle.get_out_channels())
            new_channels_num = len(torch.unique(
                head_filter_bundle.get_out_channels()))
            if new_channels_num < 1 or new_head_features_num < 1:
                #breakpoint()
                continue
            head_out_channels = new_channels_num + channel_counter - 1
            channel_counter += new_channels_num
            self.heads_out_idx.append(head_idx)
            self.head_filter_bundles.append(head_filter_bundle)
            self.heads_out_channels_num.append(new_head_features_num)
            self.heads_out_channels.append(head_out_channels)
            self.norm_layers.append( torch.nn.BatchNorm1d(new_head_features_num) )
            if reduce_to_size is not None:
                self.reduction_layers.append( torch.nn.Linear(in_features=new_head_features_num, out_features=reduce_to_size, bias=True) )
            else:
                self.reduction_layers.append(None)
            # breakpoint()

        self.head_mapping = []
        for i, head_idx in enumerate(self.heads_out_idx):
            head_out = None
            if reduce_to_size is None:
                head_out = torch.tensor([head_idx]).repeat(
                    (self.heads_out_channels_num[i])).to(self.device)
            else:
                head_out = torch.tensor([head_idx]).repeat(
                    (reduce_to_size)).to(self.device)
            self.head_mapping.append(head_out)
        # breakpoint()

    def get_head_out_channels_num(self):
        return self.heads_out_channels_num

    def get_head_out_channels_mapping(self):
        return self.heads_out_channels

    def get_heads_out_mapping(self, return_as='Tensor'):
        if return_as == 'Tensor':
            return torch.cat(self.head_mapping, dim=0).squeeze()
        return self.head_mapping

    def get_heads_out_num(self):
        return len(self.heads_out_idx)

    def forward(self, x):
        outputs = []
        activation = torch.nn.GELU()
        # Iterate over each filter and every channel in the input matrix
        for k, (filter_bundle, head_input, norm_layer, reduction_layer) in enumerate(zip(self.head_filter_bundles, self.head_inputs, self.norm_layers, self.reduction_layers)):
            # breakpoint()
            head_x = x[:, head_input]
            # breakpoint()
            # Apply the filter bundle to the selected subset of input features (to each input channel)
            # B1: filter_bundle already returns [B, F]; the removed .squeeze() used to
            # collapse [1, F] -> [F] for a size-1 batch and crash BatchNorm1d ("got 1D").
            channel = norm_layer(filter_bundle(head_x))
            if reduction_layer is not None:
                channel =  activation(reduction_layer(channel))
                #breakpoint()
            # B2: MoA gate this pathway head. z_gated = z_head * (1 + gamma * T_k). With
            # BC_MOA_GATE off, or on a non-pathway layer (_moa_gate is None), this is a no-op.
            if _MOA_GATE and self._moa_gate is not None:
                channel = channel * (1.0 + _MOA_GAMMA * self._moa_gate[k])
            outputs.append(channel)

        # Concatenate the outputs along the channel dimension.
        # B1: keep it 2D [B, C_total] -- the old trailing .squeeze() collapsed the batch
        # dim for a size-1 eval batch and broke the next hierarchy layer's x[:, idx].
        return torch.cat(outputs, dim=-1)


class CustomFilterConvLayer(nn.Module):
    def __init__(self, custom_filters, input_size=None, input_channel_mapping=None):
        super(CustomFilterConvLayer, self).__init__()

        if input_size is None and input_channel_mapping is None:
            raise Exception(
                "Neither input_channel_mapping not input_size is defined for KnowledgeBasedConvLayer. Cannot create a class instance.")

        self.input_channel_mapping = input_channel_mapping
        if self.input_channel_mapping is None:
            self.input_channel_mapping = torch.ones(input_size)
        self.output_channel_mapping = None
        # custom_filters is a list of filter modules (instances of nn.Module)
        self.filter_bundle_factory = FilterBundleFactory()
        for conv_filter in custom_filters:
            self.filter_bundle_factory.register_filter(conv_filter)

        self.channel_filter_bundles = []
        self.channel_inputs = []
        self.out_channels = []
        self.out_channels_num = 0
        print(self.input_channel_mapping)
        self.norm_layers = torch.nn.ModuleList([])

        for channel_idx in torch.unique(self.input_channel_mapping):
            channel_input = torch.where(
                self.input_channel_mapping == channel_idx, 1, 0)
            self.channel_inputs.append(channel_input)
            channel_input_size = torch.sum(channel_input)
            channel_filter_bundle = self.filter_bundle_factory.get_filter_bundle(
                channel_input_size)
            self.channel_filter_bundles.append(channel_filter_bundle)
            filter_bundle_out_channels = channel_filter_bundle.get_out_channels() + \
                self.out_channels_num
            # breakpoint()
            print(channel_input_size)
            print(channel_filter_bundle.get_out_channels())
            self.out_channels_num = torch.max(filter_bundle_out_channels)
            self.out_channels = self.out_channels + \
                [filter_bundle_out_channels]
            self.norm_layers.append( torch.nn.LazyBatchNorm1d() )
        self.out_channels = torch.cat(self.out_channels)

    def get_out_channels_num(self):
        return self.out_channels_num

    def get_out_channels_mapping(self):
        return self.out_channels

    def forward(self, x):
        outputs = []

        # Iterate over each filter and every channel in the input matrix
        for filter_bundle, channel_input, norm in zip(self.channel_filter_bundles, self.channel_inputs, self.norm_layers):

            channel_x = x[channel_input]
            # Apply the filter to the selected subset of input features (to each input channel)
            outputs.append(norm(filter_bundle(channel_x)))

        # Concatenate the outputs along the channel dimension
        return torch.cat(outputs, dim=-1).squeeze()


class HierarchicalMultiHeadModule(nn.Module):
    def __init__(self, custom_filter_classes, hierarchy_mappings, device, reduce_to_size=None, filter_head_size_param_names=None, filter_other_params=None, bottom_adjacency=None):
        # hierarchy_mappings: a list of boolean matrices that map one hierarchy level to another
        super(HierarchicalMultiHeadModule, self).__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers_names = []
        self.reduce_to_size = reduce_to_size

        input_channel_mapping = torch.ones(hierarchy_mappings[0].size()[0])
        hierarchy_layers_num = len(hierarchy_mappings)
        hierarchy_updater = None
        for i in range(hierarchy_layers_num):
            hierarchy = hierarchy_mappings[i]
            hierarchy_map = HierarchyMap(input_channel_mapping=input_channel_mapping,
                                        device=device,
                                         hierarchy_mapping=hierarchy,
                                         hierarchy_updater=hierarchy_updater)
            # breakpoint()
            hierarchy_channel_mapping = hierarchy_map.get_channel_mapping()
            # Only the bottom (gene) level carries a KEGG gene-gene graph. Pass the
            # gene symbols + adjacency + the expanded-column->gene-index map so each
            # head can slice its pathway's adjacency submatrix.
            graph_gene_symbols = graph_adjacency = graph_col2gene = None
            if i == 0 and bottom_adjacency is not None:
                graph_gene_symbols, graph_adjacency = bottom_adjacency
                graph_col2gene = hierarchy_map.hierarchy_mapping_idx
            multi_head_layer = MultiHeadLayer(
                custom_filter_classes=custom_filter_classes,
                input_head_mapping=hierarchy_channel_mapping,
                device=device,
                reduce_to_size=reduce_to_size,
                filter_head_size_param_names=filter_head_size_param_names,
                filters_other_params=filter_other_params,
                graph_gene_symbols=graph_gene_symbols,
                graph_adjacency=graph_adjacency,
                graph_col2gene=graph_col2gene)

            # breakpoint()
            self.layers.append(hierarchy_map)
            self.layers.append(multi_head_layer)
            self.layers_names.append(f"hierarchy_map_{i}")
            self.layers_names.append(f"multi_head_layer_{i}")

            hierarchy_heads_mapping = multi_head_layer.get_heads_out_mapping(
                return_as='List')
            input_channel_mapping = multi_head_layer.get_heads_out_mapping(
                return_as='Tensor')
            # head_out_num = multi_head_layer.get_heads_out_num()
            hierarchy_updater = HierarchyUpdater(
                hierarchy_heads_mapping, device)
            # hierarchy_updater.update(hierarchy_heads_mapping)

    def forward(self, x):
        #layer_counter = 0
        for layer in self.layers:
            #layer_counter += 1
            #print(layer_counter)
            #breakpoint()
            x = layer(x)
        return x


class HierarchicalConvolutionModule(nn.Module):
    def __init__(self, custom_filters, hierarchy_mappings):
        # hierarchy_mappings: a list of boolean matrices that map one hierarchy level to another
        super(HierarchicalConvolutionModule, self).__init__()
        self.layers = torch.nn.ModuleList([])
        hierarchy_updater = HierarchyUpdater(1)

        input_channel_mapping = torch.ones(hierarchy_mappings[0].size()[0])
        for hierarchy in hierarchy_mappings:
            hierarchy_map = HierarchyMap(input_channel_mapping=input_channel_mapping,
                                         hierarchy_mapping=hierarchy,
                                         hierarchy_updater=hierarchy_updater)
            hierarchy_channel_mapping = hierarchy_map.get_channel_mapping()
            conv_layer = CustomFilterConvLayer(
                custom_filters=custom_filters, input_channel_mapping=hierarchy_channel_mapping)

            self.layers.append(hierarchy_map)
            self.layers.append(conv_layer)

            input_channel_mapping = conv_layer.get_out_channels_mapping()
            conv_layer_channels_num = conv_layer.get_out_channels_num()
            hierarchy_updater.update(conv_layer_channels_num)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    """
    # Multi-layer perceptron with configurable parameters

        input_dim: The number of features in the input.
        output_dim: The number of features in the output.
        hidden_layers: A list specifying the number of units in each hidden layer.
        activation: The activation function to use ('relu', 'tanh', or 'sigmoid').
        batch_norm: Whether to use batch normalization.
        dropout: The dropout rate (0.0 means no dropout, and 1.0 would mean dropping out all units, which is not practical).
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers,
                 activation,
                 virtual_batch_size,
                 momentum=0.01,
                 batch_norm='1d',
                 dropout=0.0,
                 activation_on_output=True,
                 sparse=False):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        # Define activation function based on the argument
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'silu':
            act_fn = SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = None
            # raise ValueError("Invalid activation function.")

        prev_dim = input_dim

        for layer_dim in hidden_layers:
            # Add linear layer
            fully_connected_layer = None
            if prev_dim is None or prev_dim == -1:
                fully_connected_layer = nn.LazyLinear(layer_dim)
            else:
                fully_connected_layer = zero_module(get_linear_layer(prev_dim, layer_dim, sparse=sparse))
                initialize_non_glu(fully_connected_layer, prev_dim, layer_dim)
            self.layers.append(fully_connected_layer)

            # Optionally add batch normalization layer
            if batch_norm == '1d' and prev_dim is not None and prev_dim > 0:
                self.layers.append(
                    GBN(layer_dim, virtual_batch_size, momentum))

            # Add activation function
            if act_fn is not None:
                self.layers.append(act_fn)

            # Optionally add dropout layer
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

            prev_dim = layer_dim

        # Add output layer
        if prev_dim is None or prev_dim == -1:
            self.layers.append(nn.LazyLinear(output_dim))
        else:
            self.layers.append(zero_module(nn.Linear(prev_dim, output_dim)))
        #self.layers.append(nn.Linear(prev_dim, output_dim))
        if activation_on_output:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlockTabular(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout,
                 virtual_batch_size=256,
                 use_scale_shift_norm=False,
                 use_checkpoint=False,
                 activation='gelu',
                 sparse=False):
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_checkpoint = use_checkpoint
        self.virtual_batch_size = virtual_batch_size
        self.use_scale_shift_norm = use_scale_shift_norm

        linear = zero_module(get_linear_layer(self.input_dim, self.output_dim, sparse=sparse))
        initialize_glu(linear, self.input_dim, self.output_dim)

        act_fn = None
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'silu':
            act_fn = SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()

        if act_fn is None:
            act_fn = nn.Identity()

        self.input_layers = nn.Sequential(
            GBN(self.input_dim, virtual_batch_size=self.virtual_batch_size),

            # torch.nn.GELU(),
            # GEGLU(),
            linear,
            act_fn
        )

        self.out_layers = nn.Sequential(
            GBN(output_dim, virtual_batch_size=virtual_batch_size),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                get_linear_layer(self.output_dim, self.output_dim, sparse=sparse)
            )
        )

        self.skip_connection = nn.Identity()  # self.input_layers  # nn.Identity()

    def forward(self, x):
        h = self.input_layers(x)
        # h = self.out_layers(h)

        return h  # torch.concat((self.skip_connection(x), h), dim=-1)


class BetaRegressionLayer(nn.Module):
    def __init__(self,
                 input_dim):
        super().__init__()
        self.layer_a = nn.Linear(input_dim, 1)
        self.layer_b = nn.Linear(input_dim, 1)

    def forward(self, x):
        a = F.softplus(self.layer_a(x)).squeeze(1)
        b = F.softplus(self.layer_b(x)).squeeze(1)
        y = a/(a+b)
        return y


class ResNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers,
                 skip_levels,
                 activation='gelu',
                 dropout=0.1,
                 virtual_batch_size=128,
                 sparse=False):

        super().__init__()
        # super(ConsistencyModelTabular, self).__init__()

        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dims = hidden_layers

        # Initialize hidden dimensions
        self.input_hidden_dims = []
        self.output_hidden_dims = []

        hidden_dims = [self.input_dim] + list(hidden_dims) + [self.output_dim]
        # skip_levels = [-1] + skip_levels
        self.skip_levels = skip_levels
        for i in range(len(hidden_dims)-1):
            if skip_levels[i] != -1:
                assert skip_levels[i] < i
                self.input_hidden_dims.append(
                    hidden_dims[i]+hidden_dims[skip_levels[i]])
            # if i > 0:
            #    self.input_hidden_dims.append(hidden_dims[i]+hidden_dims[i-1])
            else:
                self.input_hidden_dims.append(hidden_dims[i])
            self.output_hidden_dims.append(hidden_dims[i+1])

        # input_channels = model_channels
        self.input_blocks = nn.ModuleList([])

        num_blocks = len(self.input_hidden_dims)
        for level in range(num_blocks):
            activation = 'gelu' if level < num_blocks-1 else None
            layers = [
                ResidualBlockTabular(self.input_hidden_dims[level],
                                     self.output_hidden_dims[level],
                                     dropout=self.dropout,
                                     activation=activation,
                                     sparse=sparse
                                     )
            ]
            # if level in attention_levels:
            #    layers.append(
            #        AttentionBlock(input_dims=self.output_hidden_dims[level],
            #                       group_dims=self.output_hidden_dims[level])
            #    )
            self.input_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Apply the model to an input batch.
        """
        # assert (y is not None) == (
        #    self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        representations = [x]

        representation = x  # .type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if self.skip_levels[i] != -1:
                representation = module(torch.concat(
                    (representation, representations[self.skip_levels[i]]), dim=-1))
            else:
                representation = module(representation)
            representations.append(representation)

        return representation  # self.out(representation)


class KEGGHierarchicalCNN(nn.Module):

    def __init__(self, device):
        super(KEGGHierarchicalCNN, self).__init__()
        from data import GeneSet
        hierarchies, kegg_genes, keggid2symbol = construct_kegg_hierarchies(GeneSet.ALL)
        # keggid_pathways2symbol(keggid2symbol)
        hierarchy_mappings = []
        for hierarchy in hierarchies:
            hierarchy_mappings.append(torch.tensor(hierarchy['weights_mask']))
        #print(list(hierarchies[0].keys()))
        # Filter operator (the KEGG "kernel") selectable via BC_FILTER for the ablation.
        _filter_map = {'conv1d': Conv1dFilter, 'linear': LinearProjectionFilter,
                       'meanpool': MeanPoolFilter, 'maxpool': MaxPoolFilter,
                       'graphconv': GraphConvFilter}
        # BC_FILTER selects ONE operator ("linear") or a COMBINATION joined by '+'
        # ("conv1d+linear", "linear+graphconv"): each head then runs all listed filters
        # and concatenates their channels (FilterBundle.forward). Combos let the dynamic
        # NN mix complementary kernels per pathway.
        _filter_name = os.environ.get('BC_FILTER', 'conv1d')
        _names = [n for n in _filter_name.split('+') if n]
        filter_classes = [_filter_map.get(n, Conv1dFilter) for n in _names]
        # For the KEGG graph convolution, hand the bottom (gene) level its real
        # gene-gene adjacency + the gene-symbol order that its input columns follow.
        bottom_adjacency = None
        if 'graphconv' in _names:
            gene_symbols = list(hierarchies[0]['from'])
            bottom_adjacency = (gene_symbols, _build_kegg_adjacency(keggid2symbol))
        self._model =  HierarchicalMultiHeadModule(
            custom_filter_classes=filter_classes,
            hierarchy_mappings=hierarchy_mappings,
            reduce_to_size=3,
            device=device,
            filter_head_size_param_names=None,  # [['kernel_size']],
            # one param dict per filter in the combination (assert len-match downstream)
            filter_other_params=[{'padding': 0, 'bias': False, 'kernel_size': -1, 'sparse': False}
                                 for _ in filter_classes],
            bottom_adjacency=bottom_adjacency,
        )
        self._device = device

        # B2: identify the pathway layer (the FIRST MultiHeadLayer, whose heads ARE the
        # KEGG pathways: layer 0 maps genes -> pathways, so head_idx == column index into
        # hierarchies[0]['to'] == the pathway name). Cache head_idx -> pathway name so a
        # per-drug target row can be aligned to the heads and gated in forward.
        self._pathway_layer = None
        self._pathway_head_names = None
        pathway_names = [str(n) for n in list(hierarchies[0]['to'])]
        for layer in self._model.layers:
            if isinstance(layer, MultiHeadLayer):
                self._pathway_layer = layer
                self._pathway_head_names = [
                    pathway_names[int(h)] if int(h) < len(pathway_names) else None
                    for h in layer.heads_out_idx]
                break

        # Auto-apply a per-process MoA target from BC_MOA_DRUG when gating is enabled, so the
        # sanctioned run_experiment.py path needs NO edits: isolate one drug per worker and
        # export BC_MOA_GATE=1 BC_MOA_DRUG=Drug_<id>.
        if _MOA_GATE:
            _d = os.environ.get('BC_MOA_DRUG', '').strip()
            if _d:
                self.apply_moa_target(_d)

    def apply_moa_target(self, drug_id=None, pathway_names=None):
        """Install the per-head MoA gate for a drug on the pathway layer.

        Pass a Drug_<id> (looked up in _MOA_DRUG_PATHWAYS) or an explicit list of KEGG
        pathway names. Builds a {0,1} vector over this encoder's pathway heads (1 where
        the head's pathway is a target of the drug) and stores it as the layer's gate.
        Matching is case-insensitive; unknown names are skipped. Passing None/empty
        clears the gate (restores ungated behavior)."""
        if self._pathway_layer is None or self._pathway_head_names is None:
            return
        if pathway_names is None:
            pathway_names = moa_target_pathways(drug_id) if drug_id else []
        wanted = {str(p).strip().lower() for p in pathway_names}
        if not wanted:
            self._pathway_layer._moa_gate = None
            return
        gate = torch.tensor(
            [1.0 if (nm is not None and nm.strip().lower() in wanted) else 0.0
             for nm in self._pathway_head_names],
            dtype=torch.float32, device=self._device)
        self._pathway_layer._moa_gate = gate

    def forward(self, x):
        return self._model(x)


class KEGGPathwayBottleneck(nn.Module):
    """Interpretable-by-construction pathway-bottleneck encoder.

    Drop-in alternative to :class:`KEGGHierarchicalCNN`. Instead of a deep
    hierarchical CNN whose gene attributions must be recovered post-hoc (via
    integrated gradients over all 7751 genes -- which failed the deletion test
    and was artifact-laden), this encoder routes genes through a single
    block-sparse gene->pathway layer. The P pathway activations z_p ARE the
    bottleneck: mechanism is read structurally off the P units, no post-hoc
    attribution needed.

    Architecture::

        x [B, 7751]
          -> masked Linear(7751 -> P)      (weight * KEGG membership mask)
          -> BatchNorm1d(P) -> GELU        == z_p, the interpretable layer
                                              (NO cross-pathway mixing yet)
          -> Linear(P -> embedding_dim)    (only place pathways may interact)
          -> embedding [B, embedding_dim]

    The [7751 x P] boolean KEGG-membership mask is registered as a
    non-trainable buffer and re-applied to the weight on every forward, so a
    gene that is not a member of pathway p can never contribute to z_p.

    Selected via the ``BC_ENCODER=pathway`` env switch (see model.py); the
    default (``hierarchical``) path is untouched.
    """

    def __init__(self, device, embedding_dim=21, min_genes=5,
                 pathways_json=None, genes_npy=None):
        super(KEGGPathwayBottleneck, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.min_genes = min_genes

        gene_symbols, pathway_names, mask = self._build_membership_mask(
            pathways_json=pathways_json, genes_npy=genes_npy,
            min_genes=min_genes)
        self.gene_symbols = gene_symbols          # order of the 7751 input cols
        self.pathway_names = pathway_names        # order of the P bottleneck units
        self.num_genes = mask.shape[0]            # 7751
        self.num_pathways = mask.shape[1]         # P (~342)

        # Non-trainable [num_genes x num_pathways] boolean membership mask.
        # Stored as float for a cheap element-wise multiply against the weight.
        self.register_buffer('membership_mask', mask.to(torch.float32))

        # Masked gene->pathway projection. Weight is [P, num_genes]; the mask is
        # [num_genes, P], so we multiply weight by mask.T on every forward.
        self.gene_to_pathway = nn.Linear(self.num_genes, self.num_pathways,
                                         bias=True)
        # Zero out non-member weights at init so the module starts exactly
        # block-sparse (masking on forward keeps it that way regardless).
        with torch.no_grad():
            self.gene_to_pathway.weight.mul_(self.membership_mask.t())

        # The interpretable per-pathway activation: BN + GELU, no mixing.
        self.pathway_norm = nn.BatchNorm1d(self.num_pathways)
        self.activation = nn.GELU()

        # The ONLY cross-pathway interaction: bottleneck -> embedding.
        self.mixing = nn.Linear(self.num_pathways, self.embedding_dim, bias=True)

        self.to(device)

    # ------------------------------------------------------------------ #
    # mask construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_membership_mask(pathways_json=None, genes_npy=None, min_genes=5):
        """Build the [num_genes x P] boolean KEGG membership mask from
        shared_data/pathways.json + shared_data/genes.npy.

        Returns (gene_symbols, pathway_names, mask_bool_tensor).
        """
        import json
        _here = os.path.dirname(os.path.abspath(__file__))
        if pathways_json is None:
            pathways_json = os.path.join(_here, 'shared_data', 'pathways.json')
        if genes_npy is None:
            genes_npy = os.path.join(_here, 'shared_data', 'genes.npy')

        with open(pathways_json) as fh:
            pathways = json.load(fh)
        genes = np.load(genes_npy, allow_pickle=True)
        gene_symbols = [str(g) for g in genes]
        gene_index = {g: i for i, g in enumerate(gene_symbols)}
        num_genes = len(gene_symbols)

        # Keep pathways with >= min_genes members present in the gene set.
        kept_names = []
        kept_members = []
        for name, members in pathways.items():
            present = [gene_index[m] for m in members if m in gene_index]
            if len(present) >= min_genes:
                kept_names.append(name)
                kept_members.append(present)

        mask = torch.zeros(num_genes, len(kept_names), dtype=torch.bool)
        for p, members in enumerate(kept_members):
            mask[members, p] = True
        return gene_symbols, kept_names, mask

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def pathway_activations(self, x):
        """Return the interpretable per-pathway activations z_p, shape [B, P].

        z_p is pathway p's score: a masked linear read of only its member
        genes, batch-normed and GELU-gated, with no cross-pathway mixing.
        """
        # Enforce block-sparsity: only member (gene, pathway) weights survive.
        masked_weight = self.gene_to_pathway.weight * self.membership_mask.t()
        pre = F.linear(x, masked_weight, self.gene_to_pathway.bias)  # [B, P]
        z = self.activation(self.pathway_norm(pre))
        return z

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.pathway_activations(x)          # [B, P] interpretable layer
        embedding = self.mixing(z)               # [B, embedding_dim]
        return embedding

    # ------------------------------------------------------------------ #
    # structural attribution
    # ------------------------------------------------------------------ #
    def pathway_jacobian(self, x, axis):
        """Pathway-level attribution: d(embedding[:, axis]) / d(z_p).

        Because the bottleneck z_p is a real layer, mechanism is read directly
        off the P pathway units -- no post-hoc gene attribution required.

        Returns a [B, P] tensor giving, per sample, the sensitivity of the
        chosen embedding axis to each pathway activation. For this architecture
        (embedding = mixing(z)) the Jacobian is the same for every sample and
        equals ``mixing.weight[axis]``; it is computed by autograd here so the
        helper stays correct if the post-bottleneck head is later deepened.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.pathway_activations(x).detach().requires_grad_(True)  # [B, P]
        out = self.mixing(z)[:, axis].sum()      # sum over batch -> per-sample grad
        grad = torch.autograd.grad(out, z, create_graph=False)[0]     # [B, P]
        return grad