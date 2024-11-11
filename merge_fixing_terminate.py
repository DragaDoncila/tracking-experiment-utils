import os
import networkx as nx
import pandas as pd
import yaml

# TODO: move this to the schema
from _iterative_solve import get_loader
from experiment_schema import Traxperiment, assign_migration_features, assign_sensitivity_features

OUT_ROOT = '/home/ddon0001/PhD/experiments/merge_terminated'
SOL_ROOT = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint_err_seg'
SCALES_PATH = '/home/ddon0001/PhD/data/cell_tracking_challenge/scales.yaml'

ds_summary = pd.read_csv(f'{SOL_ROOT}/summary.csv')
with open(SCALES_PATH, 'r') as f:
    scales = yaml.safe_load(f)

if __name__ == '__main__':
    
    for row in ds_summary.itertuples():
        ds_name = row.ds_name

        solution_path = os.path.join(SOL_ROOT, ds_name, 'matched_solution.graphml')
        node_match_path = os.path.join(SOL_ROOT, ds_name, 'matching.json')
        all_edges_pth = os.path.join(SOL_ROOT, ds_name, 'all_edges.csv')
        all_vertices_pth = os.path.join(SOL_ROOT, ds_name, 'all_vertices.csv')

        sol = nx.read_graphml(solution_path, node_type=int)
        all_edges = pd.read_csv(all_edges_pth)
        all_vertices = pd.read_csv(all_vertices_pth)

        merges = {node for node in sol.nodes if sol.in_degree(node) > 1}
        terminated = set()
        all_edges['oracle_is_correct'] = -1
        # TODO: get rid of needing this to solve from edges, make it a switch
        all_edges['learned_migration_cost'] = -1
        for merge_node_id in merges:
            merge_parents = list(sol.predecessors(merge_node_id))
            # we terminate the furthest parent of the merge vertex
            furthest_parent = max(merge_parents, key= lambda x : sol.edges[x, merge_node_id]['cost'])
            terminated.add(furthest_parent)
            all_edges.loc[(all_edges['u'] == furthest_parent) & (all_edges['v'] == -4), 'oracle_is_correct'] = 1

        if len(terminated):
            with open(os.path.join(SOL_ROOT, ds_name, 'config.yaml'), 'r') as f:
                config = yaml.load(f, Loader=get_loader())
            config['data_config']['out_root_path'] = OUT_ROOT
            config['tracktour_config']['div_constraint'] = False
            config['tracktour_config']['allow_merges'] = False

            trk_experiment = Traxperiment(**config)
            trkr = trk_experiment.as_tracker()
            tracked = trkr.solve_from_existing_edges(
                all_vertices=all_vertices,
                all_edges=all_edges,
                frame_key=trk_experiment.data_config.frame_key,
                location_keys=trk_experiment.data_config.location_keys,
                value_key=trk_experiment.data_config.value_key,
            )
            assign_migration_features(tracked.all_edges, tracked.tracked_detections)
            assign_sensitivity_features(tracked.all_edges, tracked.model)
            trk_experiment.write_solved(tracked, start=0)
            trk_experiment.evaluate(tracked.tracked_detections, tracked.tracked_edges, write_out=True)
        else:
            print(f'No new nodes to introduce for {ds_name}')