import json
import os
import networkx as nx
import pandas as pd
import yaml

from tifffile import imwrite
from tracktour import Tracker, load_tiff_frames

OUT_ROOT = '/home/ddon0001/PhD/experiments/merge_fns/'
SOL_ROOT = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint'
SCALES_PATH = '/home/ddon0001/PhD/data/cell_tracking_challenge/scales.yaml'

ds_summary = pd.read_csv(f'{SOL_ROOT}/summary.csv')
with open(SCALES_PATH, 'r') as f:
    scales = yaml.safe_load(f)

EDGE_FN = 'EdgeFlag.FALSE_NEG'
EDGE_FP = 'EdgeFlag.FALSE_POS'
EDGE_WS = 'EdgeFlag.WRONG_SEMANTIC'

NODE_FN = 'NodeFlag.FALSE_NEG'
NODE_FP = 'NodeFlag.FALSE_POS'
NODE_NS = 'NodeFlag.NON_SPLIT'

if __name__ == '__main__':
    
    for row in ds_summary.itertuples():
        ds_name = row.ds_name

        gt_solution_path = os.path.join(SOL_ROOT, ds_name, 'matched_gt.graphml')
        solution_path = os.path.join(SOL_ROOT, ds_name, 'matched_solution.graphml')
        node_match_path = os.path.join(SOL_ROOT, ds_name, 'matching.json')

        sol = nx.read_graphml(solution_path, node_type=int)
        gt = nx.read_graphml(gt_solution_path)
        with open(node_match_path) as f:
            node_match = json.load(f)
        gt_to_sol = {item[0]: item[1] for item in node_match}
        sol_to_gt = {item[1]: item[0] for item in node_match}

        merges = {node for node in sol.nodes if sol.in_degree(node) > 1}
        found_gts = set()
        if len(scales[ds_name]['pixel_scale']) == 3:
            new_vinfo = {
                't': [],
                'z': [],
                'y': [],
                'x': [],
                'label': [],
                'gt_label': []
            }
            is_3d = True
        else:
            new_vinfo = {
                't': [],
                'y': [],
                'x': [],
                'label': [],
                'gt_label': []
            }
            is_3d = False

        next_label = max([sol.nodes[node]['label'] for node in sol.nodes]) + 1
        for merge_node_id in merges:
            merge_parents = list(sol.predecessors(merge_node_id))
            for parent in merge_parents:
                if sol.edges[parent, merge_node_id][EDGE_FP]:
                    gt_node_id = sol_to_gt[parent]
                    true_successors = list(gt.successors(gt_node_id))
                    assert len(true_successors) <= 1, 'Ground truth parent is dividing!'
                    if len(true_successors)\
                        and gt.nodes[(true_successor := true_successors[0])][NODE_FN]\
                        and true_successor not in found_gts:
                        found_gts.add(true_successor)
                        true_successor_info = gt.nodes[true_successor]
                        new_vinfo['t'].append(true_successor_info['t'])
                        new_vinfo['label'].append(next_label)
                        new_vinfo['y'].append(true_successor_info['y'])
                        new_vinfo['x'].append(true_successor_info['x'])
                        if is_3d:
                            new_vinfo['z'].append(true_successor_info['z'])
                        new_vinfo['gt_label'].append(true_successor_info['segmentation_id'])
                        next_label += 1
        if len(found_gts):
            out_path = os.path.join(OUT_ROOT, ds_name)
            out_seg = os.path.join(out_path, 'SEG')
            os.makedirs(out_seg, exist_ok=True)

            # need to repair segmentation here by copying over the introduced vertices
            # into the segmentation, and saving it
            orig_seg = load_tiff_frames(row.seg_path)
            gt_seg = load_tiff_frames(row.tra_gt_path)

            for i in range(len(found_gts)):
                gt_label = new_vinfo['gt_label'][i]
                new_label = new_vinfo['label'][i]
                frame = new_vinfo['t'][i]
                label_mask = gt_seg[frame] == gt_label
                orig_seg[frame][label_mask] = new_label
            n_digits = max(len(str(len(orig_seg))), 3)
            for i, frame in enumerate(orig_seg):
                frame_out_name = os.path.join(out_seg, f"mask{str(i).zfill(n_digits)}.tif")
                imwrite(frame_out_name, frame, compression="zlib")

            # then remove 'gt_label' from v_info
            new_vinfo.pop('gt_label')
            to_introduce = pd.DataFrame(new_vinfo)

            original_det = pd.read_csv(row.det_path)[to_introduce.columns]
            new_det = pd.concat([original_det, to_introduce], ignore_index=True)
            im_shape = eval(row.im_shape)
            scale = scales[ds_name]['pixel_scale']

            tracker = Tracker(im_shape=im_shape, scale=scale)
            tracker.DEBUG_MODE = True
            tracker.MERGE_EDGE_CAPACITY = 1
            tracker.USE_DIV_CONSTRAINT = False
            tracker.PENALIZE_FLOW = False
            tracker.ALLOW_MERGES = False

            tracked = tracker.solve(
                detections=new_det,
                frame_key='t',
                location_keys=['z', 'y', 'x'] if is_3d else ['y', 'x'],
            )
            # write detections and edges
            tracked.tracked_detections.to_csv(os.path.join(out_path, 'tracked_detections.csv'), index=False)
            tracked.tracked_edges.to_csv(os.path.join(out_path, 'tracked_edges.csv'), index=False)
            # write nx graph
            sol_graph = tracked.as_nx_digraph()
            nx.write_graphml(sol_graph, os.path.join(out_path, 'nx_sol.graphml'))
        else:
            print(f'No new nodes to introduce for {ds_name}')