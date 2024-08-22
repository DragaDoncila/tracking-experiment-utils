import json
import os
import pandas as pd


with_merge_root = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint'
no_merge_root = '/home/ddon0001/PhD/experiments/misc/no_merges'
ds_summary = pd.read_csv(os.path.join(with_merge_root, 'summary.csv'))

ds_names = []
old_aogms = []
new_aogms = []
aogm_diffs = []
old_tras = []
new_tras = []
tra_diffs = []
for row in ds_summary.itertuples():
    ds_name = row.ds_name
    with_merge_metrics_pth = os.path.join(with_merge_root, ds_name, 'metrics.json')
    no_merge_metrics_pth = os.path.join(no_merge_root, ds_name, 'metrics.json')
    with open(with_merge_metrics_pth, 'r') as f:
        with_merge_metrics = json.load(f)
        old_aogm = with_merge_metrics['AOGM']
        old_tra = with_merge_metrics['TRA']
    with open(no_merge_metrics_pth, 'r') as f:
        no_merge_metrics = json.load(f)
        new_aogm = no_merge_metrics['AOGM']
        new_tra = no_merge_metrics['TRA']
    aogm_diffs.append(new_aogm - old_aogm)
    old_aogms.append(old_aogm)
    new_aogms.append(new_aogm)
    ds_names.append(ds_name)
    old_tras.append(old_tra)
    new_tras.append(new_tra)
    tra_diffs.append(new_tra - old_tra)

metrics_compare_df = pd.DataFrame({
    'ds_name': ds_names,
    'old_aogm': old_aogms,
    'new_aogm': new_aogms,
    'aogm_diff': aogm_diffs,
    'old_tra': old_tras,
    'new_tra': new_tras,
    'tra_diff': tra_diffs
})
print(metrics_compare_df)