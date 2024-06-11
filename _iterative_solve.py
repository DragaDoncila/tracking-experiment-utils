
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

def iterate_solution(detections_path, initial_solution_path):
    det_df = pd.read_csv(detections_path)

    all_vertices = pd.read_csv(join_pth(initial_solution_path, OUT_FILE.ALL_VERTICES), index_col=0)
    all_edges = pd.read_csv(join_pth(initial_solution_path, OUT_FILE.ALL_EDGES))
    solution_graph = nx.read_graphml(join_pth(initial_solution_path, OUT_FILE.MATCHED_SOL), node_type=int)
    gt_graph = nx.read_graphml(join_pth(initial_solution_path, OUT_FILE.MATCHED_GT))

    with open(join_pth(initial_solution_path, OUT_FILE.MATCHING), 'r') as f:
        node_match = json.load(f)
    gt_to_sol = {item[0]: item[1] for item in node_match}
    sol_to_gt = {item[1]: item[0] for item in node_match}

    with open(join_pth(initial_solution_path, OUT_FILE.CONFIG), 'r') as f:
        config = yaml.load(f, Loader=get_loader())
    # assign features to all migration edges in all_edges and on solution graph
    # NOTE: we assign to all migration-like edges in solution, regardless of 
    # whether node is dividing
    assign_migration_features(all_edges, det_df)
    
    all_edges['sampled'] = 0
    all_edges['oracle_is_correct'] = -1
    solution_edges = all_edges[all_edges.flow > 0]
    first_sample_ids, oracle_labels = get_first_sample_ids(solution_edges, solution_graph)

    update_with_sample(all_edges, first_sample_ids, oracle_labels, gt_graph, sol_to_gt)
    train_and_predict_migrations(all_edges)
    assign_new_costs(all_edges)

    trkr = Traxperiment(**config).as_tracker()


    tracked = trkr.solve_from_existing_edges(
        all_vertices=all_vertices,
        all_edges=all_edges,
        frame_key=config['data_config']['frame_key'],
        location_keys=config['data_config']['location_keys'],
        value_key=config['data_config']['value_key'],
        k_neighbours=config['instance_config']['k']
    )



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

def get_first_sample_ids(
        solution_edges,
        solution_graph,
        gt_graph,
        sol_to_gt_matching,
        n_init=10,
    ):
    migration_rows = solution_edges[solution_edges.chosen_neighbour_rank >= 0]
    # we sample using the chosen_neighbour_distance as a weight
    # makes it more likely we get an incorrect edge
    chosen_sample = migration_rows.sample(n=n_init, weights='distance')
    is_mig_correct = [
        # NOTE: we only exclude FALSE POSITIVE EDGES here
        # wrong semantic edges ARE PART OF THE SOLUTION so we don't want to remove them...
        not (solution_graph.edges[e['u'], e['v']]['EdgeFlag.FALSE_POS'])
        # solution_graph.edges[e['u'], e['v']]['EdgeFlag.WRONG_SEMANTIC'])
        for _, e in chosen_sample.iterrows()
        ]
    wrong_edges = len(chosen_sample) - np.sum(is_mig_correct)
    if not wrong_edges:
        # are all cells dividing
        gt_sources = [sol_to_gt_matching[u] for u in chosen_sample.u if u in sol_to_gt_matching]
        if all(gt_graph.out_degree(u) > 1 for u in gt_sources):
            # all cells are dividing and all edges are correct. Bad sample!
            raise ValueError("Couldn't sample any guaranteed incorrect edges", chosen_sample)

    # #TODO: could infinite, need better here
    # while not wrong_edges:
    #     print("Resampling!")
    #     chosen_sample = migration_rows.sample(n=n_init, weights='distance')
    #     is_mig_correct = [
    #             not (solution_graph.edges[e['u'], e['v']]['EdgeFlag.FALSE_POS'])
    #             # NOTE: we only exclude FALSE POSITIVE EDGES here
    #             # wrong semantic edges ARE PART OF THE SOLUTION so we don't want to remove them...
    #             # solution_graph.edges[e['u'], e['v']]['EdgeFlag.WRONG_SEMANTIC'])
    #             for _, e in chosen_sample.iterrows()
    #         ]
    #     wrong_edges = len(chosen_sample) - np.sum(is_mig_correct)
    # is_mig_correct = np.asarray(is_mig_correct, dtype=int)
    return chosen_sample.index, is_mig_correct

def update_with_sample(
        all_edges,
        sample_ids,
        oracle_labels,
        gt_graph,
        sol_to_gt_matching,
    ):
    all_edges.loc[sample_ids, 'sampled'] = 1
    all_edges.loc[sample_ids, 'oracle_is_correct'] = oracle_labels
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