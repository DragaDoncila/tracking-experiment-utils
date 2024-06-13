
"""
Assume: tracktour has already been run once WITH DEBUG MODE 
and the detections and all edges have been saved.

Starting Point:
Input: path to detections, path to initial solution

- load initial solution
- detections path has area of each node
- need to extract the features of migrations (and all_edges)
- ask for first sample (!)
- train model on sample, predict on all_edges
- save everything....
- assign costs for migrations in the all_edges df
- build tracker from all_edges and solve again
- save everything
- repeat until everything has been sampled???
"""
import json
import networkx as nx
import os
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import yaml
from experiment_schema import OUT_FILE, Cost, Traxperiment

def iterate_solution(detections_path, initial_solution_path, out_root_path):
    det_df = pd.read_csv(detections_path)
    with open(join_pth(initial_solution_path, OUT_FILE.CONFIG), 'r') as f:
        config = yaml.load(f, Loader=get_loader())

    all_vertices, all_edges, solution_graph, gt_graph, gt_to_sol, sol_to_gt = load_sol_files(initial_solution_path)

    # assign features to all migration edges in all_edges and on solution graph
    # NOTE: we assign to all migration-like edges in solution, regardless of 
    # whether node is dividing
    assign_migration_features(all_edges, det_df)
    # assign new out root to config, give it a new ds name
    config['data_config']['out_root_path'] = out_root_path
    all_edges['sampled_it'] = -1
    it = 1
    sampling_strategy = generate_sampling_strategy(solution_graph.number_of_edges())
    new_sample_ids, oracle_labels = get_first_sample_ids(all_edges, solution_graph, gt_graph, sol_to_gt, n_init=sampling_strategy[0])
    for n_samples in sampling_strategy[1:]:
        ds_name = config['data_config']['dataset_name']
        if it > 1:
            ds_name = ds_name.rstrip('0123456789') + f'{it}'
        else:
            ds_name += '_it1'
        config['data_config']['dataset_name'] = ds_name
        update_with_sample(all_edges, it, new_sample_ids, oracle_labels, gt_graph, sol_to_gt)
        train_and_predict_migrations(all_edges)
        assign_new_costs(all_edges)

        trk_experiment = Traxperiment(**config)
        trkr = trk_experiment.as_tracker()
        tracked = trkr.solve_from_existing_edges(
            all_vertices=all_vertices,
            all_edges=all_edges,
            frame_key=config['data_config']['frame_key'],
            location_keys=config['data_config']['location_keys'],
            value_key=config['data_config']['value_key'],
            k_neighbours=config['instance_config']['k']
        )
        trk_experiment.write_solved(tracked, start=0)
        results, matched = trk_experiment.evaluate(tracked.tracked_detections, tracked.tracked_edges, write_out=True)
        
        # need to reload/reassign all new stuff here
        all_vertices, all_edges, solution_graph, gt_graph, gt_to_sol, sol_to_gt = load_sol_files(os.path.join(out_root_path, config['data_config']['dataset_name']))
        new_sample_ids, oracle_labels = get_new_sample_ids(all_edges, solution_graph, n_samples)
        it += 1

def load_sol_files(root_pth):
    all_vertices = pd.read_csv(join_pth(root_pth, OUT_FILE.ALL_VERTICES), index_col=0)
    if all_vertices.index.name == 't':
        all_vertices = all_vertices.reset_index()
    all_edges = pd.read_csv(join_pth(root_pth, OUT_FILE.ALL_EDGES))
    solution_graph = nx.read_graphml(join_pth(root_pth, OUT_FILE.MATCHED_SOL), node_type=int)
    gt_graph = nx.read_graphml(join_pth(root_pth, OUT_FILE.MATCHED_GT))

    with open(join_pth(root_pth, OUT_FILE.MATCHING), 'r') as f:
        node_match = json.load(f)
    gt_to_sol = {item[0]: item[1] for item in node_match}
    sol_to_gt = {item[1]: item[0] for item in node_match}
    return all_vertices, all_edges, solution_graph, gt_graph, gt_to_sol, sol_to_gt

def load_cost_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Cost:
    return Cost.INTERCHILD_DISTANCE

def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor('tag:yaml.org,2002:python/object/apply:experiment_schema.Cost', load_cost_constructor)
    return loader

def join_pth(root: str, suffix: OUT_FILE) -> str:
    return os.path.join(root, suffix.value)

def assign_migration_features(
        all_edges,
        det_df    
    ): 
    all_edges['chosen_neighbour_rank'] = -1
    all_edges['chosen_neighbour_area_prop'] = -1

    migration_edges = all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)]
    for name, group in migration_edges.groupby('u'):
        sorted_group = group.sort_values(by='distance').reset_index()
        for i, row in enumerate(sorted_group.itertuples()):
            u, v = row.u, row.v
            u_area = det_df.loc[u, 'area']
            v_area = det_df.loc[v, 'area']
            all_edges.loc[row.index, 'chosen_neighbour_area_prop'] = v_area / u_area
            all_edges.loc[row.index, 'chosen_neighbour_rank'] = i

def get_migration_correct_labels(chosen_sample, solution_graph):
    is_correct = [
    # NOTE: we only exclude FALSE POSITIVE EDGES here
    # wrong semantic edges ARE PART OF THE SOLUTION so we don't want to remove them...
    not (solution_graph.edges[e['u'], e['v']]['EdgeFlag.FALSE_POS'])
    # solution_graph.edges[e['u'], e['v']]['EdgeFlag.WRONG_SEMANTIC'])
    for _, e in chosen_sample.iterrows()
    ]
    return is_correct

def generate_sampling_strategy(n):
    sample_sizes = []

    # First 1% of n with 30 equispaced samples
    first_1_percent = int(0.01 * n)
    first_sample_size = first_1_percent / 30
    if first_sample_size < 10:
        first_sample_size = 10
        sample_sizes.extend([10 for _ in range(first_sample_size, first_1_percent, 10)])
    else:                            
        sample_sizes.extend([first_sample_size for _ in range(first_sample_size, first_1_percent, first_sample_size)])

    # From 1% to 10% of n, sampling 1% of n each iteration
    for i in range(1, 10):
        sample_sizes.append(int(0.01 * n))

    # From 10% to 100% of n, sampling every 10%
    for i in range(1, 11):
        sample_sizes.append(int(0.1 * n))

    return sample_sizes

def get_first_sample_ids(
        all_edges,
        solution_graph,
        gt_graph,
        sol_to_gt_matching,
        n_init=10,
    ):
    all_edges['sampled'] = 0
    all_edges['oracle_is_correct'] = -1
    all_edges['manually_repaired'] = -1
    solution_edges = all_edges[all_edges.flow > 0]
    migration_rows = solution_edges[solution_edges.chosen_neighbour_rank >= 0]
    # we sample using the chosen_neighbour_distance as a weight
    # makes it more likely we get an incorrect edge
    chosen_sample = migration_rows.sample(n=n_init, weights='distance')
    is_mig_correct = get_migration_correct_labels(chosen_sample, solution_graph)
    wrong_edges = len(chosen_sample) - np.sum(is_mig_correct)
    if not wrong_edges:
        # are all cells dividing
        gt_sources = [sol_to_gt_matching[u] for u in chosen_sample.u if u in sol_to_gt_matching]
        if all(gt_graph.out_degree(u) > 1 for u in gt_sources):
            # all cells are dividing and all edges are correct. Bad sample!
            raise ValueError("Couldn't sample any guaranteed incorrect edges", chosen_sample)

    is_mig_correct = np.asarray(is_mig_correct, dtype=int)
    return chosen_sample.index, is_mig_correct

def get_new_sample_ids(
        all_edges,
        solution_graph,
        n_samples=10
    ):
    solution_edges = all_edges[all_edges.flow > 0]
    migration_edges = solution_edges[solution_edges.chosen_neighbour_rank >= 0]
    # unsampled: sampled flag 0, used in the solution, non-virtual adjacent
    unsampled_edges = migration_edges[(migration_edges.sampled == 0)]
    assert all(unsampled_edges.mig_predict_proba != -1), 'Some edges have no migration prediction!'
    # sample using the migration prediction probability as weight
    to_sample = n_samples if len(unsampled_edges) >= n_samples else len(unsampled_edges)
    if to_sample == 0:
        return [], []
    
    unsampled_edges['sample_weight'] = 1 - unsampled_edges.mig_predict_proba
    chosen_sample = unsampled_edges.sample(n=to_sample, weights='sample_weight')
    is_mig_correct = get_migration_correct_labels(chosen_sample, solution_graph)
    is_mig_correct = np.asarray(is_mig_correct, dtype=int)
    return chosen_sample.index, is_mig_correct

def update_with_sample(
        all_edges,
        iteration,
        sample_ids,
        oracle_labels,
        gt_graph,
        sol_to_gt_matching,
    ):
    all_edges.loc[sample_ids, 'sampled'] = 1
    all_edges.loc[sample_ids, 'oracle_is_correct'] = oracle_labels
    all_edges.loc[sample_ids, 'sampled_it'] = iteration
    # NOTE: if the edge is correct, we check whether source node is dividing (in GT!!)
    # if it's not, we mark all other migrations from that node as incorrect
    for idx, is_correct in zip(sample_ids, oracle_labels):
        if is_correct:
            u, v = all_edges.loc[idx, ['u', 'v']]
            if u in sol_to_gt_matching:
                u_gt = sol_to_gt_matching[u]
                # gt source is not dividing, current edge is only correct edge
                if gt_graph.out_degree(u_gt) <= 1:
                    # mark all other migrations incorrect
                    other_mig_from_u = all_edges[(all_edges.u == u) & (all_edges.v >= 0) & (all_edges.v != v)]
                    all_edges.loc[other_mig_from_u.index, 'sampled'] = 1
                    all_edges.loc[other_mig_from_u.index, 'oracle_is_correct'] = 0
                    all_edges.loc[other_mig_from_u.index, 'sampled_it'] = iteration
                    # if any of them had flow > 0, mark them as manually repaired
                    were_flowing = other_mig_from_u[other_mig_from_u.flow > 0]
                    if not were_flowing.empty: 
                        all_edges.loc[were_flowing.index, 'manually_repaired'] = 1

def train_and_predict_migrations(all_edges, n_estimators=250, max_depth=10):
    all_edges['mig_predict_proba'] = -1
    columns_of_interest = [
        'distance',
        'chosen_neighbour_area_prop',
        'chosen_neighbour_rank',
        'oracle_is_correct'
    ]
    all_sampled = all_edges[all_edges.sampled == 1][columns_of_interest]

    X = all_sampled.drop(columns=['oracle_is_correct'])
    y = all_sampled.oracle_is_correct

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X, y)

    all_migration_edges = all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)][columns_of_interest].drop(columns=['oracle_is_correct'])
    mig_predictions = rf.predict_proba(all_migration_edges)
    true_class = list(rf.classes_).index(1)
    prob_correct = mig_predictions[:, true_class]
    all_edges.loc[all_migration_edges.index, 'mig_predict_proba'] = prob_correct

def prob_weighted_cost(row):
    return row.distance * (1 - row.mig_predict_proba)

def assign_new_costs(all_edges):
    all_edges['learned_migration_cost'] = -1
    edges_with_pred = all_edges[all_edges.mig_predict_proba >= 0]
    all_edges.loc[edges_with_pred.index, 'learned_migration_cost'] = edges_with_pred.apply(prob_weighted_cost, axis=1)