import pandas as pd

from experiment_schema import TraxData, TraxInstance, TraxTour, Traxperiment
from utils import get_scale

def ds_summary_to_configs(ds_summary_path, out_root, appearance_cheat=False, div_constraint=True, merge_capacity=2):
    """Convert dataset summary to Traxperiment configs.

    Parameters
    ----------
    ds_summary_path : str
        path to dataset summary csv
    out_root : str
        directory to save output to
    appearance_cheat : bool, optional
        appearance cheat, by default False
    div_constraint : bool, optional
        whether to use explicit div constraint, by default True
    merge_capacity : int, optional
        how many cells can merge into one for an extended path, by default 2
        
    Returns
    -------
    List[Traxperiment]
        list of Traxperiment configs
    """
    ds_summary = pd.read_csv(ds_summary_path)
    configs = []
    for _, row in ds_summary.iterrows():
        frame_shape = eval(row['im_shape'])
        data_config = TraxData(
            dataset_name=row['ds_name'],
            detections_path=row['det_path'],
            frame_shape=frame_shape,
            scale=get_scale(row['ds_name']),
            frame_key='t',
            location_keys=["y", "x"] if len(frame_shape) == 2 else ["z", "y", "x"],
            image_path=row['im_path'],
            segmentation_path=row['seg_path'],
            out_root_path=out_root,
            ground_truth_path=row['tra_gt_path'],
            value_key='label'
        )
        instance_config = TraxInstance()
        tracktour_config = TraxTour(
            appearance_cheat=appearance_cheat,
            div_constraint=div_constraint,
            merge_capacity=merge_capacity,
        )
        exp_config = Traxperiment(
            data_config=data_config,
            instance_config=instance_config,
            tracktour_config=tracktour_config
        )
        configs.append(exp_config)

    return configs