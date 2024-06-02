import pandas as pd
import numpy as np
import json
import os
import time
from generate_configs import get_config_for_row

baseline_root = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint'
ds_summary = pd.read_csv(f'{baseline_root}/summary.csv')
super_unitary = pd.read_csv(f'{baseline_root}/superunitary_counts.csv')
out_root = '/home/ddon0001/PhD/experiments/grid_again/'

# we're going to try a range of flow penalties
# a couple sub 1, some small numbers up to 20
# then every 10 up to 100
search_space = np.concatenate((
    np.arange(0.1, 1.01, 0.1),
    np.arange(2, 20.01, 2),
    np.arange(30, 100.01, 10)
))
for _, row in ds_summary.iterrows():
    ds_out_res = os.path.join(out_root, 'summaries', f"{row['ds_name']}_final.csv")
    if os.path.exists(ds_out_res):
        print("#" * 50)
        print("Search complete for ", row['ds_name'])
        print("#" * 50)
        continue
    if row['ds_name'] == 'Fluo-C3DH-H157_02':
        print("#" * 50)
        print("Skipping ", row['ds_name'])
        print("#" * 50)
        continue
    if super_unitary[super_unitary.ds_name == row['ds_name']]['num_super_edges'].values[0] == 0:
        print("#" * 50)
        print("Skipping ", row['ds_name'], " as it has no super unitary edges")
        print("#" * 50)
        continue
    print("#" * 50)
    print("Searching for ", row['ds_name'])
    print("#" * 50)
    searched = []
    aogm = []
    ds_name = row['ds_name']
    metrics_pth = os.path.join(baseline_root, ds_name, 'metrics.json')
    with open(metrics_pth, 'r') as f:
        metrics = json.load(f)
        baseline_metric = metrics['AOGM']
    if baseline_metric == 0:
        print("#" * 50)
        print(f'Skipping {ds_name} as baseline AOGM is 0')
        print("#" * 50)
        continue
    last_best = baseline_metric
    for penalty in search_space:
        # load config
        config = get_config_for_row(
            row,
            out_root,
            div_constraint=False,
            penalize_flow=True,
            flow_penalty=penalty
        )
        # solve
        start = time.time()
        tracked = config.run(write_out=False)
        # evaluate
        results, matched = config.evaluate(tracked.tracked_detections, tracked.tracked_edges, write_out=False)
        # is it better than baseline?
        if results['AOGM'] < last_best:
            print(f'Found improvement for {ds_name} with penalty {penalty}')
            last_best = results['AOGM']
            # save config, results, matched
            config.data_config.dataset_name = f'{ds_name}_{penalty}'
            config.write_solved(tracked, start)
            config.write_metrics(results, matched)
        searched.append(penalty)
        aogm.append(results['AOGM'])
    summary = pd.DataFrame({'penalty': searched, 'AOGM': aogm})
    summary.to_csv(ds_out_res, index=False)
    print("#" * 50)
    print("Search complete for ", ds_name)
    print("#" * 50)