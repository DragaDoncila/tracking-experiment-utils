"""
Given a root directory containing cell tracking challenge formatted datasets,
extract detections and generate a summary csv with information about each dataset.

NB: if a `detections.csv` file is already present for a dataset, the extraction
step is skipped.

Sored in csv:
- ds_name: Dataset name & sequence: "{name}_{sequence}"
- im_shape: Image shape as tuple of integers
- n_frames: Number of frames
- min_det: Minimum number of detections per frame
- max_det: Maximum number of detections per frame
- total_det: Total number of detections
"""
import glob
import os
import warnings

import pandas as pd
from tqdm import tqdm
from tracktour import get_im_centers, load_tiff_frames


IM_DIR_GLOB = "[0-9][0-9]/"
ST_SEG_SUFFIX = "_ST/SEG/"
GT_TRA_SUFFIX = "_GT/TRA/"

DET_CSV_NAME = 'detections.csv'

TIME_KEY = 't'

    
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def generate_ctc_summary(root_dir, ds_summary_path):
    """Write summary info of ctc datasets in root_dir to ds_summary_path.

    Parameters
    ----------
    root_dir : str
        path to directory containing CTC datasets
    ds_summary_path : str
        path to write summary csv

    Returns
    -------
    pd.DataFrame
        summary dataframe
    """
    # get all folders and use basename as dataset name
    ds_names = get_immediate_subdirectories(root_dir)
    if not ds_names:
        raise ValueError(f"Given directory has no subdirectories: {root_dir}")
    rows = []
    for ds_name in tqdm(ds_names):
        ds_im_path = os.path.join(root_dir, ds_name, IM_DIR_GLOB)
        im_seqs = glob.glob(ds_im_path)
        if not im_seqs:
            warnings.warn(f"No image sequences found for dataset {ds_name}. Skipping dataset...", UserWarning)
            continue
        for seq_pth in im_seqs:
            # load ims get shape and number of frames
            seq = os.path.basename(os.path.dirname(seq_pth)).split('_RES')[0]
            in_seg_path = os.path.join(root_dir, ds_name, f'{seq}{ST_SEG_SUFFIX}')
            out_det_path = os.path.join(in_seg_path, DET_CSV_NAME)

            gt_path = os.path.join(root_dir, ds_name, f'{seq}{GT_TRA_SUFFIX}')
            if not os.path.exists(gt_path):
                raise ValueError(f"Ground truth path not found: {gt_path}")

            name = f'{ds_name}_{seq}'
            ims = load_tiff_frames(seq_pth)
            im_shape = ims[0].shape
            n_frames = ims.shape[0]
            if not os.path.exists(out_det_path):
                # extract detections
                _, detections, _, _, _ = get_im_centers(in_seg_path)
                detections.to_csv(out_det_path)
            else:
                warnings.warn(f"Found existing detections for {name}. Using them...", UserWarning)
                detections = pd.read_csv(out_det_path)
            grouped_det = detections.groupby(TIME_KEY).size()
            min_cells = grouped_det.min()
            max_cells = grouped_det.max()
            total_cells = len(detections)
            rows.append(
                {
                    'ds_name': name,
                    'im_shape': im_shape,
                    'n_frames': n_frames,
                    'min_det': min_cells,
                    'max_det': max_cells,
                    'total_det': total_cells,
                    'det_path': out_det_path,
                    'im_path': seq_pth,
                    'seg_path': in_seg_path,
                    'tra_gt_path': gt_path
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ds_summary_path)
    return df