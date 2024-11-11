import json
import networkx as nx
import os
from tracktour import load_tiff_frames
from traccuracy import TrackingGraph
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

GT_ROOT = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/'
SOL_ROOT = '/home/ddon0001/PhD/experiments/merge_fns/'

ds_names = sorted(os.listdir(SOL_ROOT))
for ds_name in ds_names:
    ds, seq = ds_name.split('_')
    gt_path = os.path.join(GT_ROOT, ds, f'{seq}_GT/TRA')
    
    sol_path = os.path.join(SOL_ROOT, ds_name, 'nx_sol.graphml')
    seg_path = os.path.join(SOL_ROOT, ds_name, 'SEG')

    seg_ims = load_tiff_frames(seg_path)
    sol_graph = nx.read_graphml(sol_path, node_type=int)
    
    sol_tracking_graph = TrackingGraph(
        sol_graph,
        label_key='label',
        location_keys = ['z', 'y', 'x'] if len(seg_ims.shape) == 4 else ['y', 'x'],
        segmentation=seg_ims,
    )
    gt_tracking_graph = load_ctc_data(gt_path)
    matcher = CTCMatcher()
    matched = matcher.compute_mapping(gt_tracking_graph, sol_tracking_graph)
    results = CTCMetrics().compute(matched)
    with open(os.path.join(SOL_ROOT, ds_name, 'metrics.json'), 'w') as f:
        json.dump(results.results, f)
    with open(os.path.join(SOL_ROOT, ds_name, 'matching.json'), 'w') as f:
        json.dump(matched.mapping, f)
    nx.write_graphml(matched.pred_graph.graph, os.path.join(SOL_ROOT, ds_name, 'matched_sol.graphml'))
    nx.write_graphml(matched.gt_graph.graph, os.path.join(SOL_ROOT, ds_name, 'matched_gt.graphml'))
    print('Done', ds_name)


    
