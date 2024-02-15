from enum import Enum, auto
import json
import time
import pandas as pd
from importlib.metadata import version
from pydantic import BaseModel, Field, conlist
from tracktour import Tracker
from typing import Optional, Union
import os

import yaml

PathLike = Union[str, bytes, os.PathLike]

class OUT_FILE(Enum):
    # solutions
    TRACKED_DETECTIONS = 'tracked_detections.csv'
    TRACKED_EDGES = 'tracked_edges.csv'
    ALL_VERTICES = 'all_vertices.csv'
    ALL_EDGES = 'all_edges.csv'

    # models
    MODEL_LP = 'model.lp'
    MODEL_MPS = 'model.mps'

    # tracktour version
    TRACKTOUR_VERSION = 'version.txt'
    
    # timings
    TIMING = 'times.json'

    # config
    CONFIG = 'config.yaml'


class TraxData(BaseModel):
    # these should also be optional, we only need them for metrics
    # maybe we should have a separate metrics config
    image_path: PathLike
    segmentation_path: PathLike
    out_root_path: PathLike
    detections_path: PathLike
    frame_shape: conlist(int, min_length=2, max_length=3)

    # Default values available
    dataset_name: str = 'cells'
    location_keys: conlist(str, min_length=2, max_length=3) = ['y', 'x']
    frame_key: str = 't'

    # Optional
    ground_truth_path: Optional[PathLike] = None
    value_key: Optional[str] = None

class TraxInstance(BaseModel):
    # migration_only: bool = False
    k: int = 10

class Cost(Enum):
    INTERCHILD_DISTANCE = auto()

class TraxTour(BaseModel):
    # pre_refactor: bool = False
    appearance_cheat: bool = False
    div_constraint: bool = True
    # min should be 1 (no merges)
    merge_capacity: int = 2
    div_cost: Cost = Cost.INTERCHILD_DISTANCE

class Traxperiment(BaseModel):
    data_config: TraxData
    instance_config: TraxInstance = Field(default_factory=TraxInstance)
    tracktour_config: TraxTour = Field(default_factory=TraxTour)

    def as_tracker(self):
        tracker = Tracker(
            im_shape=tuple(self.data_config.frame_shape), 
            k_neighbours=self.instance_config.k
        )
        tracker.DEBUG_MODE = True

        # configure merge capacity
        tracker.MERGE_EDGE_CAPACITY = self.tracktour_config.merge_capacity
        # if appearance cheat, change appearance capacity to 2
        if self.tracktour_config.appearance_cheat:
            tracker.APPEARANCE_EDGE_CAPACITY = 2
        # TODO: configure cost
        # TODO: configure div constraint
        
        return tracker

    def run(self):
        start = time.time()
        tracker = self.as_tracker()
        
        # load detections
        detections = pd.read_csv(self.data_config.detections_path)
        
        # solve model
        tracked = tracker.solve(
            detections=detections,
            frame_key=self.data_config.frame_key,
            location_keys=self.data_config.location_keys
        )

        self.write_solved(tracked, start)

        return tracked
    
    def write_solved(self, tracked, start):
        # write everything out (including tracktour commit...)
        out_root = self.data_config.out_root_path
        # TODO: formalize/document that we're writing out and potentially overwriting path
        out_ds = os.path.join(out_root, self.data_config.dataset_name)
        os.makedirs(out_ds, exist_ok=True)
        
        # write out dataframes
        tracked_dict = dict(tracked)
        for k in [OUT_FILE.TRACKED_EDGES, OUT_FILE.TRACKED_DETECTIONS, OUT_FILE.ALL_EDGES, OUT_FILE.ALL_VERTICES]:
            file_key = k.value
            attr_key = file_key.split('.')[0]
            out_pth = self.get_path_to_out_file(k)
            if (info := tracked_dict.pop(attr_key)) is not None:
                info.to_csv(out_pth, index=False)

        # write out model (mps and lp)
        if (model := tracked_dict.pop('model')) is not None:
            model.write(self.get_path_to_out_file(OUT_FILE.MODEL_LP))
            model.write(self.get_path_to_out_file(OUT_FILE.MODEL_MPS))
        
        # TODO: formalize/document that the BUILT version is written, so we don't want to
        # be running experiments in editable mode
        # write tracktour version
        tracktour_version = version(distribution_name='tracktour')
        with open(self.get_path_to_out_file(OUT_FILE.TRACKTOUR_VERSION), 'w') as f:
            f.write(tracktour_version)

        # write config (yaml ideally)
        with open(self.get_path_to_out_file(OUT_FILE.CONFIG), 'w') as f:
            yaml.dump(self.model_dump(), f)

        # experiment wall time
        duration = time.time() - start
        tracked_dict.update({
            'exp_time': duration
        })
        with open(self.get_path_to_out_file(OUT_FILE.TIMING), 'w') as f:
            json.dump(tracked_dict, f)
    
    def get_path_to_out_file(self, file_key: OUT_FILE):
        out_ds = os.path.join(self.data_config.out_root_path, self.data_config.dataset_name)
        return os.path.join(out_ds, file_key.value)

if __name__ == '__main__':
    DATA_ROOT = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/'
    data_config = TraxData(
        dataset_name='Fluo-N2DL-HeLa_01',
        image_path=os.path.join(DATA_ROOT, '01/'),
        segmentation_path=os.path.join(DATA_ROOT, '01_ST/SEG/'),
        detections_path='/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/01_ST/SEG/detections.csv',
        out_root_path='/home/ddon0001/PhD/experiments/fix_capacity/',
        frame_shape=[700, 1100]
    )
    experiment = Traxperiment(data_config=data_config)
    experiment.tracktour_config.appearance_cheat = True
    
    tracked = experiment.run()
    migration_edges = tracked.tracked_edges
    tracked_detections = tracked.tracked_detections