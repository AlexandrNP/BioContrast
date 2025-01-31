import os
import numpy as np
from data import ResponseDataloadersFactory, Source, GeneSet, get_pathways
from Bio.KEGG.REST import *
from data import DATA_DIR


from torch import nn


def fetch_brite_hierarchy(top_entry="br:hsa00001"):
    # Fetch the BRITE hierarchy for pathways (ko00001 is an example of a BRITE hierarchy ID)
    brite_hierarchy_raw = kegg_get(top_entry).read()
    out_filename = top_entry
    out = open(out_filename, 'w')
    out.write(brite_hierarchy_raw)
    out.close()

    # This will hold the hierarchy
    hierarchy = {}
    current_main_category = None
    current_subcategory = None

    # Process the BRITE file line by line
    for line in brite_hierarchy_raw.split("\n"):
        line = line.lstrip(' \t')
        if line.startswith("A"):
            # Main category (e.g., "A09100 Metabolism")
            current_main_category = line[1:].strip()
            hierarchy[current_main_category] = {}
        elif line.startswith("B"):
            # Subcategory (e.g., "B09101 Carbohydrate metabolism")
            current_subcategory = line[1:].strip()
            if current_main_category:
                hierarchy[current_main_category][current_subcategory] = []
        elif line.startswith("C"):
            # Actual pathway entry under a subcategory
            if current_main_category and current_subcategory:
                line = line[1:].strip()
                pathway_id, pathway_name = line.split(" ", 1)
                hierarchy[current_main_category][current_subcategory].append(
                    pathway_id.strip())
                #   (pathway_id.strip(), pathway_name.strip()))

    return hierarchy

# ko_link = kegg_link("pathway", "ko").read()
# ko_db = ko_link.split('\n')


def merge_dicts(dictionaries):
    result = {}
    for dictionary in dictionaries:
        result.update(dictionary)

    return result


def hierarchies_to_layers(hierarchies, layers=None, include_values=False):
    if layers is None:
        layers = []

    # Reshuffling hierarchies and collecting all values in a layer
    all_values = []
    is_next_level = True
    matrices = {}
    next_hierarchies = []
    for key in hierarchies:
        values = None
        if type(hierarchies[key]) == dict:
            values = list(hierarchies[key].keys())
            next_hierarchies.append(hierarchies[key])
        elif include_values:
            is_next_level = False
            values = list(hierarchies[key])
        else:
            is_next_level = False
            continue
        all_values += values
        if key not in matrices:
            matrices[key] = []
        matrices[key].append(values)

    # Construct a layer
    layer_from = sorted(np.unique(all_values))
    layer_to = sorted(list(matrices.keys()))
    weights = []
    for key in layer_to:
        filter = np.zeros(len(layer_from))
        _, indices, _ = np.intersect1d(
            layer_from, matrices[key], return_indices=True)
        filter[indices] = 1
        weights.append(filter)
    weights = np.array(weights).T

    out = {
        'from': layer_from,
        'to': layer_to,
        'weights_mask': weights
    }
    new_layer = [out]
    if is_next_level and len(next_hierarchies) > 0:
        next_level = merge_dicts(next_hierarchies)
        layers = hierarchies_to_layers(next_level, layers, include_values)

    return new_layer + layers


def pathways_names2ids(kegg_pathways_genes, kegg_pathways_ids):
    id_mapping = {}
    for key in kegg_pathways_genes:
        pathway_id = kegg_pathways_ids[key][3:]
        # print(key, pathway_id)
        id_mapping[pathway_id] = kegg_pathways_genes[key]
    return id_mapping


def intersect_neighboring_layers(layer_prev, layer_next, update_next_only=False):
    # Correspondence is the following:
    #   layer_prev.from -> inputs
    #   layer_prev.to -> intermediates <- layer_next.from  | intersection happens here
    #   layer_next.to -> outputs
    intersection, indices_prev, indices_next = np.intersect1d(
        layer_prev['to'], layer_next['from'], return_indices=True)
    if not update_next_only:
        layer_prev['to'] = np.array(layer_prev['to'])[indices_prev]
        layer_prev['weights_mask'] = layer_prev['weights_mask'][:, indices_prev]
    layer_next['from'] = np.array(layer_next['from'])[indices_next]
    layer_next['weights_mask'] = layer_next['weights_mask'][indices_prev, :]
    return layer_prev, layer_next


def write_gene_list(gene_list, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as out:
        for gene in gene_list:
            out.write(gene)


def construct_kegg_hierarchies():
    use_pathways_names = True
    kegg_pathways = get_pathways()
    kegg_dict = {value[3:]: key for key, value in kegg_pathways[1].items()}
    pathways = pathways_names2ids(kegg_pathways[0], kegg_pathways[1])
    pathway_hierarchy = fetch_brite_hierarchy()
    hierarchy_layers = hierarchies_to_layers(
        pathway_hierarchy, include_values=True)
    hierarchy_layers.reverse()
    gene_layer = hierarchies_to_layers(pathways, include_values=True)
    hierarchy_layers = gene_layer + hierarchy_layers

    # Substitute pathway codes with pathway names
    if use_pathways_names:
        for i in range(len(hierarchy_layers)):
            for layer in ['from', 'to']:
                if hierarchy_layers[i][layer][0] in kegg_dict:
                    # breakpoint()
                    hierarchy_layers[i][layer] = np.array(
                        [kegg_dict[x] for x in hierarchy_layers[i][layer] if x in kegg_dict])

    # Make sure that all entries from the current layer match the entries from the previous layer
    for i in range(len(hierarchy_layers)-1):
        layer_prev, layer_next = intersect_neighboring_layers(
            hierarchy_layers[i], hierarchy_layers[i+1])
        hierarchy_layers[i] = layer_prev
        hierarchy_layers[i+1] = layer_next

    write_gene_list(hierarchy_layers[0]['from'], 'KEGG')
    return hierarchy_layers

    # dataloader_factory = ResponseDataloadersFactory(
    #    cell_line_source=Source.CCLE,
    #    gene_set=GeneSet.ONCOGENES_REG,
    #    cross_validation_num=5,
    #    validation_size=0.12,
    #    random_seed=2023,
    #    device='cuda:6')


if __name__ == "__main__":
    layers = construct_kegg_hierarchies()
    print(list(layers[0].keys()))
    breakpoint()
