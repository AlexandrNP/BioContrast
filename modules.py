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
#import sparselinear as sl


def get_linear_layer(in_features, out_features, sparse=False, bias=True):
    #if sparse:
    #    return sl.SparseLinear(in_features, out_features, bias=bias, sparsity=0.5)
    #else:
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
        self.batched_kernel = vmap(
            lambda batch, weights=self.weights: weights(batch))
        self.initialized = True

    def get_out_features_num(self):
        print(f"In features: {self.in_features}")
        print(f"Kernel size: {self.kernel_size}")
        print(f"Out features: {self.out_features}")
        return self.out_features

    def forward(self, x):
        if not self.initialized:
            raise Exception(
                f"An instance of the class {type(self)} has to be initialized before usage! This class has a lazy initialization")
        if self.padding > 0:
            #breakpoint()
            x = torch.concat((torch.zeros(x.shape[0], self.padding).to(device=self.device), x, torch.zeros(x.shape[0], self.padding).to(device=self.device)), dim=-1)
        preprocessed_x = self._preprocess(x.squeeze())
        preprocessed_x = preprocessed_x.unfold(1, self.kernel_size, 1)
        return self.batched_kernel(preprocessed_x).squeeze()


class LinearProjectionFilter(FilterBase):
    def __init__(self, out_features_as_fraction=None, out_features_num=None, bias=True):
        super(LinearProjectionFilter, self).__init__()
        if out_features_as_fraction is not None and out_features_num is not None:
            raise Exception("All features")
        self.out_features


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
            outputs.append(filter_output)

        # Collect output from all channels and return
        return torch.cat(outputs, dim=0).squeeze()


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
        #breakpoint()
        return x[:,self.hierarchy_mapping_idx].squeeze()


class MultiHeadLayer(nn.Module):
    def __init__(self, custom_filter_classes, filter_head_size_param_names, input_head_mapping, device='cpu', filters_other_params=None, reduce_to_size=None):
        super(MultiHeadLayer, self).__init__()

        self.device = device
        self.input_head_mapping = input_head_mapping
        self.output_channel_mapping = None
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

            self.filter_bundle_factory = FilterBundleFactory()

            for conv_filter_class, filter_params, head_size_names in zip(custom_filter_classes, filter_param_list, filter_head_size_param_names):
                for name in head_size_names:
                    filter_params[name] = head_input_size
                current_filter_params = deepcopy(filter_params)
                current_filter_params['device'] = self.device
                if 'kernel_size' in current_filter_params:
                    if current_filter_params['kernel_size'] == -1:
                        kernel_size = head_input_size // 2 #min(5, head_input_size // 2)
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
        for filter_bundle, head_input, norm_layer, reduction_layer in zip(self.head_filter_bundles, self.head_inputs, self.norm_layers, self.reduction_layers):
            # breakpoint()
            head_x = x[:, head_input]
            # breakpoint()
            # Apply the filter bundle to the selected subset of input features (to each input channel)
            channel = norm_layer(filter_bundle(head_x).squeeze())
            if reduction_layer is not None:
                channel =  activation(reduction_layer(channel))
                #breakpoint()
            outputs.append(channel)

        # Concatenate the outputs along the channel dimension
        return torch.cat(outputs, dim=-1).squeeze()


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
    def __init__(self, custom_filter_classes, hierarchy_mappings, device, reduce_to_size=None, filter_head_size_param_names=None, filter_other_params=None):
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
            multi_head_layer = MultiHeadLayer(
                custom_filter_classes=custom_filter_classes,
                input_head_mapping=hierarchy_channel_mapping,
                device=device,
                reduce_to_size=reduce_to_size,
                filter_head_size_param_names=filter_head_size_param_names,
                filters_other_params=filter_other_params)

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
        conv1d_filter_class = Conv1dFilter
        filter_classes = [conv1d_filter_class]
        self._model =  HierarchicalMultiHeadModule(
            custom_filter_classes=filter_classes,
            hierarchy_mappings=hierarchy_mappings,
            reduce_to_size=3,
            device=device,
            filter_head_size_param_names=None,  # [['kernel_size']], 
            filter_other_params=[{'padding': 0, 'bias': False, 'kernel_size': -1, 'sparse': False}]
        )
    
    def forward(self, x):
        return self._model(x)