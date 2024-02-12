from enum import Enum, auto
import pandas as pd
from pydantic import BaseModel, Field, conlist
from tracktour import Tracker
from typing import Optional, Union, List
import os

PathLike = Union[str, bytes, os.PathLike]

class TraxData(BaseModel):
    # these should also be optional, we only need them for metrics
    # maybe we should have a separate metrics config
    image_path: PathLike
    segmentation_path: PathLike
    out_root_path: PathLike
    detections_path: PathLike
    frame_shape: conlist(int, min_length=2, max_length=3)

    # Default values
    dataset_name: str = 'cells'
    location_keys: conlist(str, min_length=2, max_length=3) = ['y', 'x']
    frame_key: str = 't'

    # Optional
    ground_truth_path: Optional[PathLike] = None
    value_key: Optional[str] = None

class TraxInstance(BaseModel):
    # migration_only: bool = False
    k: int = 10

# class TraxTracker(BaseModel):
#     pass
    
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
    # tracker_config: TraxTracker
    tracktour_config: TraxTour = Field(default_factory=TraxTour)

    def as_tracker(self):
        tracker = Tracker(
            im_shape=tuple(self.data_config.frame_shape), 
            k_neighbours=self.instance_config.k
        )

        # configure merge capacity
        tracker.MERGE_EDGE_CAPACITY = self.tracktour_config.merge_capacity
        # if appearance cheat, change appearance capacity to 2
        if self.tracktour_config.appearance_cheat:
            tracker.APPEARANCE_EDGE_CAPACITY = 2
        # TODO: configure cost
        # TODO: configure div constraint
        
        return tracker

    def run(self):
        tracker = self.as_tracker()
        
        # load detections
        detections = pd.read_csv(self.data_config.detections_path)
        
        migration_edges = tracker.solve(
            detections=detections,
            frame_key=self.data_config.frame_key,
            location_keys=self.data_config.location_keys
        )
        return migration_edges

if __name__ == '__main__':
    DATA_ROOT = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/'
    data_config = TraxData(
        dataset_name='Fluo-N2DL-HeLa_1',
        image_path=os.path.join(DATA_ROOT, '01/'),
        segmentation_path=os.path.join(DATA_ROOT, '01_ST/SEG/'),
        detections_path='/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/01_ST/SEG/detections.csv',
        out_root_path='/home/ddon0001/PhD/experiments/fix_capacity/',
        frame_shape=[700, 1100]
    )
    experiment = Traxperiment(data_config=data_config)
    
    print(experiment.model_dump_json(indent=2))
    migration_edges = experiment.run()
    print(migration_edges.head())