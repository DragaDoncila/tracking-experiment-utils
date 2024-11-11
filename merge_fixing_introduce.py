"""Introduce the furthest parent of each merge vertex into the current frame.

Save the updated segmentation and detection files in OUT_ROOT ready for solving
and re-evaluating. To be used with `merge_fixing_introduce_pipeline.ipynb`.
"""
import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from skimage.measure import regionprops_table
from tifffile import imwrite
from tracktour import Tracker, load_tiff_frames

OUT_ROOT = '/home/ddon0001/PhD/data/cell_tracking_challenge/merge_introduced_data'
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

        sol = nx.read_graphml(solution_path, node_type=int)

        merges = {node for node in sol.nodes if sol.in_degree(node) > 1}
        introduced = set()
        if len(scales[ds_name]['pixel_scale']) == 3:
            new_vinfo = {
                't': [],
                'z': [],
                'y': [],
                'x': [],
                'label': [],
                'parent_label': [],
                'parent_t': []
            }
            is_3d = True
        else:
            new_vinfo = {
                't': [],
                'y': [],
                'x': [],
                'label': [],
                'parent_label': [],
                'parent_t': []
            }
            is_3d = False

        next_label = max([sol.nodes[node]['label'] for node in sol.nodes]) + 1
        for merge_node_id in merges:
            merge_parents = list(sol.predecessors(merge_node_id))
            # we introduce a node by copying the location/cell of the most distant parent
            # into the current frame

            furthest_parent = max(merge_parents, key= lambda x : sol.edges[x, merge_node_id]['cost'])
            introduced.add(furthest_parent)
            to_introduce_info = sol.nodes[furthest_parent]
            new_vinfo['t'].append(sol.nodes[merge_node_id]['t'])
            new_vinfo['label'].append(next_label)
            new_vinfo['y'].append(to_introduce_info['y'])
            new_vinfo['x'].append(to_introduce_info['x'])
            if is_3d:
                new_vinfo['z'].append(to_introduce_info['z'])
            new_vinfo['parent_t'].append(to_introduce_info['t'])
            new_vinfo['parent_label'].append(to_introduce_info['label'])
            next_label += 1

        if len(introduced):
            ds, seq = ds_name.split('_')
            out_path = os.path.join(OUT_ROOT, ds)
            out_seg = os.path.join(out_path, f'{seq}_ERR_SEG')
            os.makedirs(out_seg, exist_ok=True)

            # need to repair segmentation here by copying over the introduced vertices
            # into the segmentation, and saving it
            orig_seg = load_tiff_frames(row.seg_path)
            new_vinfo['area'] = []
            for i in range(len(introduced)):
                parent_label = new_vinfo['parent_label'][i]
                parent_frame = new_vinfo['parent_t'][i]
                new_label = new_vinfo['label'][i]
                new_frame = new_vinfo['t'][i]
                label_mask = orig_seg[parent_frame] == parent_label
                orig_seg[new_frame][label_mask] = new_label
                new_vinfo['area'].append(regionprops_table(label_mask.astype(np.uint8), properties=['area'])['area'][0])
            n_digits = max(len(str(len(orig_seg))), 3)
            for i, frame in enumerate(orig_seg):
                frame_out_name = os.path.join(out_seg, f"mask{str(i).zfill(n_digits)}.tif")
                imwrite(frame_out_name, frame, compression="zlib")

            # then remove 'gt_label' from v_info
            new_vinfo.pop('parent_label')
            new_vinfo.pop('parent_t')
            to_introduce = pd.DataFrame(new_vinfo)

            # need to copy over area and other det info
            original_det = pd.read_csv(row.det_path)[to_introduce.columns]
            new_det = pd.concat([original_det, to_introduce], ignore_index=True)
            new_det.to_csv(os.path.join(out_seg, 'detections.csv'), index=False)
        else:
            print(f'No new nodes to introduce for {ds_name}')