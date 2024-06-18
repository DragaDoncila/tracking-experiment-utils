
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

    os.makedirs(out_root_path, exist_ok=True)
    with open(os.path.join(out_root_path, 'sampling_strategy.txt'), 'w') as f:
        for n_sampled in sampling_strategy:
            f.write(f'{n_sampled}\n')

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
        new_sample_ids, oracle_labels = get_new_sample_ids_random(all_edges, solution_graph, n_samples)
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

    # first 10 iterations are 30 edges each - ~10 min annotation effort
    n_30s = min(10, n // 30)
    sample_sizes.extend([30 for _ in range(n_30s)])
    n -= n_30s * 30
    
    # next 10 iterations are 180 edges each - ~1 hour annotation effort
    n_180s = min(10, n // 180)
    sample_sizes.extend([180 for _ in range(n_180s)])
    n -= n_180s * 180

    # finally, we evenly divide what's left into 10 iterations?
    final_sample_size = max(180, n // 10)
    final_n_samples = n // final_sample_size
    sample_sizes.extend([final_sample_size for _ in range(final_n_samples)])
    n -= final_n_samples * final_sample_size

    # remainder
    if n > 0:
        if len(sample_sizes):
            sample_sizes[-1] += n
        else:
            sample_sizes.append(n)
    return sample_sizes

def generate_sampling_strategy_old(n):
    sample_sizes = []

    # First 1% of n with 30 equispaced samples
    first_1_percent = int(0.01 * n)
    first_sample_size = int(first_1_percent / 30)
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

def get_new_sample_ids_proba_weighted(
        all_edges,
        solution_graph,
        n_samples=10
    ):

    unsampled_edges = get_unsampled_edges(all_edges)
    assert all(unsampled_edges.scaled_mig_predict_proba != -1), 'Some edges have no migration prediction!'
    to_sample = n_samples if len(unsampled_edges) >= n_samples else len(unsampled_edges)
    if to_sample == 0:
        return [], []
    
    # sample using the scaled migration prediction probability as weight
    # the more likely it is an edge is incorrect, the less likely we are to sample it
    unsampled_edges['sample_weight'] = 1 - unsampled_edges.scaled_mig_predict_proba
    # scaled mig predict, should be no zero weights
    assert len(unsampled_edges[unsampled_edges.sample_weight == 0]) == 0, 'Some unsampled edges have zero weight!'

    chosen_sample = unsampled_edges.sample(n=to_sample, weights='sample_weight')
    is_mig_correct = get_migration_correct_labels(chosen_sample, solution_graph)
    is_mig_correct = np.asarray(is_mig_correct, dtype=int)
    return chosen_sample.index, is_mig_correct

def get_new_sample_ids_random(
        all_edges,
        solution_graph,
        n_samples=10
    ):
    unsampled_edges = get_unsampled_edges(all_edges)
    to_sample = n_samples if len(unsampled_edges) >= n_samples else len(unsampled_edges)
    if to_sample == 0:
        return [], []
    
    # no weighted sampling, totally random
    chosen_sample = unsampled_edges.sample(n=to_sample)
    is_mig_correct = get_migration_correct_labels(chosen_sample, solution_graph)
    is_mig_correct = np.asarray(is_mig_correct, dtype=int)
    return chosen_sample.index, is_mig_correct

def get_unsampled_edges(all_edges):
    # unsampled: sampled flag 0, used in the solution, non-virtual adjacent
    solution_edges = all_edges[all_edges.flow > 0]
    migration_edges = solution_edges[solution_edges.chosen_neighbour_rank >= 0]
    unsampled_edges = migration_edges[(migration_edges.sampled == 0)]
    return unsampled_edges

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

def train_and_predict_migrations(all_edges, n_estimators=250, max_depth=10, k=0.9):
    """Train random forest classifier on sampled migrations and predict on all edges.

    Scale model's prediction probability using k as a measure of confidence. k=0 will
    make all probabilities 0.5, while k=1 will keep the original model probability.

    Parameters
    ----------
    all_edges : pd.DataFrame
        dataframe of all edges
    n_estimators : int, optional
        number of trees, by default 250
    max_depth : int, optional
        max depth of trees, by default 10
    k : float, optional
        model probability scale factor, by default 0.9
    """
    all_edges['mig_predict_proba'] = -1
    all_edges['scaled_mig_predict_proba'] = -1
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
    all_edges.loc[all_migration_edges.index, 'scaled_mig_predict_proba'] = prob_correct * k + (1 - k) * 0.5

def prob_weighted_cost_old(row):
    return row.distance * (1 - row.mig_predict_proba)

def scaled_prob_weighted_cost(row):
    """Designed to use scaled probabilities closer to 0.5 as model tends to be too confident."""
    return row.distance * (1 - row.scaled_mig_predict_proba)

def assign_new_costs(all_edges):
    all_edges['learned_migration_cost'] = -1
    edges_with_pred = all_edges[all_edges.scaled_mig_predict_proba >= 0]
    all_edges.loc[edges_with_pred.index, 'learned_migration_cost'] = edges_with_pred.apply(scaled_prob_weighted_cost, axis=1)