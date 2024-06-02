import json
import networkx as nx
import os
import pandas as pd
import yaml

def get_scale(
    ds_name, 
    scale_config_path='/home/ddon0001/PhD/data/cell_tracking_challenge/scales.yaml'
):
    with open(scale_config_path, 'r') as f:
        scales = yaml.load(f, Loader=yaml.FullLoader)
    return scales[ds_name]['pixel_scale']

def classify_all_divisions(root_path, write_out=False):
    datasets = pd.read_csv(os.path.join(root_path, 'summary.csv'))
    ds_names = []
    count_total = []
    div_proper = []
    div_cheating = []
    div_immediate_merge = []
    div_prior_merge = []
    div_super_merge = []
    div_super_parent = []

    all_proper = {}
    all_immediate_merges = {}
    all_prior_merges = {}
    all_cheats = {}
    all_super_merges = {}
    all_super_parents = {}

    for _, row in datasets.iterrows():
        div_sol_path = os.path.join(root_path, row['ds_name'], 'matched_solution.graphml')
        div_sol_all_edges_path = os.path.join(root_path, row['ds_name'], 'all_edges.csv')

        div_sol = nx.read_graphml(div_sol_path, node_type=int)
        all_edges = pd.read_csv(div_sol_all_edges_path)
        parent_nodes = [node for node in div_sol.nodes if div_sol.out_degree(node) > 1]

        counts, indices = classify_divisions(row['ds_name'], div_sol, all_edges, parent_nodes)

        count_total.append(counts['total'])
        ds_names.append(row['ds_name'])
        div_proper.append(counts['proper'])
        div_immediate_merge.append(counts['immediate_merge'])
        div_prior_merge.append(counts['prior_merge'])
        div_super_merge.append(counts['super_merge'])
        div_cheating.append(counts['cheating'])
        div_super_parent.append(counts['super_parent'])

        all_cheats[row['ds_name']] = indices['cheating']
        all_prior_merges[row['ds_name']] = indices['prior_merge']
        all_super_merges[row['ds_name']] = indices['super_merge']
        all_super_parents[row['ds_name']] = indices['super_parent']
        all_immediate_merges[row['ds_name']] = indices['immediate_merge']
        all_proper[row['ds_name']] = indices['proper']


    new_div_df = pd.DataFrame({
        'ds_name': ds_names,
        'new_proper': div_proper,
        'new_immediate_merge': div_immediate_merge,
        'new_cheating': div_cheating,
        'new_prior_merge': div_prior_merge,
        # we have just two across all datasets
        'new_super_merge': div_super_merge,
        'new_super_parent': div_super_parent,
        'total': count_total
    })
    if write_out:
        with open(os.path.join(root_path, 'cheats.json'), 'w') as f:
            json.dump(all_cheats, f)
        with open(os.path.join(root_path, 'prior_merges.json'), 'w') as f:
            json.dump(all_prior_merges, f)
        with open(os.path.join(root_path, 'super_merges.json'), 'w') as f:
            json.dump(all_super_merges, f)
        with open(os.path.join(root_path, 'super_parents.json'), 'w') as f:
            json.dump(all_super_parents, f)
        with open(os.path.join(root_path, 'immediate_merges.json'), 'w') as f:
            json.dump(all_immediate_merges, f)
        with open(os.path.join(root_path, 'proper.json'), 'w') as f:
            json.dump(all_proper, f)
    all_dict = {
        'cheating': all_cheats,
        'prior_merges': all_prior_merges,
        'super_merges': all_super_merges,
        'super_parents': all_super_parents,
        'immediate_merges': all_immediate_merges,
        'proper': all_proper
    }
    return new_div_df, all_dict

def classify_divisions(ds_name, div_sol, all_edges, parent_nodes):
    count_total = len(parent_nodes)

    count_cheating = 0
    count_proper = 0
    count_immediate_merge = 0
    count_prior_merge = 0
    count_super_merge = 0
    count_super_parent = 0

    proper = []
    prior_merges = []
    cheating = []
    super_merges = []
    super_parents = []
    immediate_merges = []
    for parent in parent_nodes:
        children = list(div_sol.successors(parent))
        has_div_flow = len(flow := all_edges[(all_edges.u == -3) & (all_edges.v == parent)]['flow'].values) and flow[0] == 1
        # sum flow from all real predecessors into parent is greater than 1, so it's a merge
        predecessors = list(div_sol.predecessors(parent))
        pred_flow = sum([div_sol.edges[predecessor, parent]['flow'] for predecessor in predecessors])
        if len(children) > 2:
            count_super_parent += 1
            super_parents.append(parent)
            continue
        if len(children) != 2:
            raise ValueError(f"How many children..? {ds_name} parent:{parent} children:{children}")
        # 1 div flow into parent means it's a "proper" division
        if has_div_flow:
            count_proper += 1
            proper.append(parent)
            continue
        if pred_flow > 1:
            # more than one predecessor for parent means we just merged
            if len(predecessors) > 1:
                count_immediate_merge += 1
                immediate_merges.append(parent)
            # just one predecessor for parent means we merged in prior frames
            else:
                # follow flow back to where it merged
                v = parent
                preds = list(div_sol.predecessors(v))
                flow = div_sol.edges[preds[0], v]['flow']
                while flow == 2:
                    v = preds[0]
                    preds = list(div_sol.predecessors(v))
                    flow = div_sol.edges[preds[0], v]['flow']
                n_parents = len(preds)
                # three parents merging for an eventual split
                if n_parents > 2:
                    super_merges.append(parent)
                    count_super_merge += 1
                # no parents or 1 parent means we should have division flow
                if n_parents == 0 or n_parents == 1:
                    # check that D is flowing into v (could be alongisde A OR a real parent)
                    if sum(all_edges[(all_edges.u == -3) & (all_edges.v == v)]['flow']) != 1:
                        raise ValueError(f"No div flow? {ds_name} v:{v} parents:{preds}")
                    count_cheating += 1
                    cheating.append(parent)
                # we have 2+ real ancestors
                else:
                    # assert n_parents == 2
                    count_prior_merge += 1
                    prior_merges.append(parent)
            continue
        raise (f"Unresolved division! {ds_name} parent:{parent}")
    counts = {
        'total': count_total,
        'proper': count_proper,
        'cheating': count_cheating,
        'immediate_merge': count_immediate_merge,
        'prior_merge': count_prior_merge,
        'super_merge': count_super_merge,
        'super_parent': count_super_parent
    }
    indices = {
        'proper': proper,
        'cheating': cheating,
        'immediate_merge': immediate_merges,
        'prior_merge': prior_merges,
        'super_merge': super_merges,
        'super_parent': super_parents
    }
    return counts, indices