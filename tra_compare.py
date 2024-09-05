import json
import os
import pandas as pd
from ctc_metrics.scripts.evaluate import evaluate_sequence

our_format_root = '/home/ddon0001/PhD/experiments/misc/no_merges'
ctc_format_root = '/home/ddon0001/PhD/experiments/as_ctc/no_merges'
gt_root = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/'

ds_summary = pd.read_csv(os.path.join(our_format_root, 'summary.csv'))

ds_names = []
our_tras = []
ctc_tras = []
tra_diffs = []
for row in ds_summary.itertuples():
    ds_name = row.ds_name
    our_metrics_path = os.path.join(our_format_root, ds_name, 'metrics.json')
    with open(our_metrics_path, 'r') as f:
        our_metrics = json.load(f)
        our_tra = our_metrics['TRA']
    
    ds, seq = ds_name.split('_')
    ctc_ds_path = os.path.join(ctc_format_root, ds, f'{seq}_RES')
    gt_ds_path = os.path.join(gt_root, ds, f'{seq}_GT')
    ctc_tra = evaluate_sequence(ctc_ds_path, gt_ds_path, ['TRA'])["TRA"]
    tra_diff = our_tra - ctc_tra
    ds_names.append(ds_name)
    our_tras.append(our_tra)
    ctc_tras.append(ctc_tra)
    tra_diffs.append(tra_diff)

metrics_df = pd.DataFrame({
    'ds_name': ds_names,
    'our_tra': our_tras,
    'ctc_tra': ctc_tras,
    'tra_diff': tra_diffs
})
print(metrics_df)



