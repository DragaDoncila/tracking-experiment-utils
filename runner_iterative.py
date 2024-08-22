import os

import pandas as pd
from _iterative_solve import iterate_solution

root_pth = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint/'
ds_summary_path = os.path.join(root_pth, 'summary.csv')
ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'det_path']]

out_root = '/home/ddon0001/PhD/experiments/iterative_misc/'

ds_name = 'Fluo-N2DL-HeLa_02'
ds_out_root = os.path.join(out_root, ds_name)

det_pth = ds_info[ds_info.ds_name == ds_name]['det_path'].values[0]
sol_pth = os.path.join(root_pth, ds_name)
iterate_solution(det_pth, sol_pth, ds_out_root)