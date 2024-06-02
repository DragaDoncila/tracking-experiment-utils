from enum import Enum, auto
import json
import networkx as nx
import time
import pandas as pd
from importlib.metadata import version
from pydantic import BaseModel, Field, conlist
from tracktour import Tracker, load_tiff_frames
from traccuracy import TrackingGraph
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics
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

    # metrics
    METRICS = 'metrics.json'
    MATCHING = 'matching.json'
    MATCHED_SOL = 'matched_solution.graphml'
    MATCHED_GT = 'matched_gt.graphml'



class TraxData(BaseModel):
    detections_path: PathLike
    # TODO: remove? Read fron image if it's present?
    frame_shape: conlist(int, min_length=2, max_length=3)
    scale: conlist(float, min_length=2, max_length=3)
    #TODO: these should also be optional, we only need them for metrics
    # maybe we should have a separate metrics config
    image_path: PathLike
    segmentation_path: PathLike
    out_root_path: PathLike

    # Default values available
    dataset_name: str = 'cells'
    location_keys: conlist(str, min_length=2, max_length=3) = ['y', 'x']
    frame_key: str = 't'

    # Optional (needed for evaluation)
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
    penalize_flow: bool = False
    # min should be 1 (no merges)
    merge_capacity: int = 2
    div_cost: Cost = Cost.INTERCHILD_DISTANCE
    #TODO: should only be used in conjunction with penalize flow
    flow_penalty: float = 0

class Traxperiment(BaseModel):
    data_config: TraxData
    instance_config: TraxInstance = Field(default_factory=TraxInstance)
    tracktour_config: TraxTour = Field(default_factory=TraxTour)

    def as_tracker(self):
        tracker = Tracker(
            im_shape=tuple(self.data_config.frame_shape),
            scale=self.data_config.scale
        )
        tracker.DEBUG_MODE = True

        # configure merge capacity
        tracker.MERGE_EDGE_CAPACITY = self.tracktour_config.merge_capacity
        # if appearance cheat, change appearance capacity to 2
        if self.tracktour_config.appearance_cheat:
            tracker.APPEARANCE_EDGE_CAPACITY = 2
        # TODO: configure div constraint
        if not self.tracktour_config.div_constraint:
            tracker.USE_DIV_CONSTRAINT = False
        if self.tracktour_config.penalize_flow:
            tracker.PENALIZE_FLOW = True
            tracker.FLOW_PENALTY_COEFFICIENT = self.tracktour_config.flow_penalty
        # TODO: configure cost
        
        return tracker

    def run(self, write_out=True):
        start = time.time()
        tracker = self.as_tracker()
        if tracker.PENALIZE_FLOW and tracker.FLOW_PENALTY_COEFFICIENT == 0:
            raise ValueError("Cannot penalize flow with a coefficient of 0.")
        
        # load detections
        detections = pd.read_csv(self.data_config.detections_path)
        
        # solve model
        tracked = tracker.solve(
            detections=detections,
            frame_key=self.data_config.frame_key,
            location_keys=self.data_config.location_keys,
            k_neighbours=self.instance_config.k
        )
        
        if write_out:
            self.write_solved(tracked, start)

        return tracked
    
    def evaluate(self, tracked_detections=None, tracked_edges=None, write_out=True):
        if self.data_config.ground_truth_path is None:
            raise ValueError(f"Cannot evaluate experiment without ground truth path configured.")
        if self.data_config.value_key is None:
            raise ValueError("Cannot evaluate experiment without `value_key` configured.")
        track_graph = self.load_solution_for_metrics(tracked_detections, tracked_edges)
        # load ground truth data using CTC loader
        gt_graph = load_ctc_data(self.data_config.ground_truth_path)

        matcher = CTCMatcher()
        matched = matcher.compute_mapping(gt_graph, track_graph)
        results = CTCMetrics().compute(matched)
        
        if write_out:
            self.write_metrics(results.results, matched)
        return results.results, matched

    def write_metrics(self, results, matched):
        out_ds = self.get_path_to_out_dir()
        with open(os.path.join(out_ds, OUT_FILE.METRICS.value), 'w') as f:
            json.dump(results, f)
        with open(os.path.join(out_ds, OUT_FILE.MATCHING.value), 'w') as f:
            json.dump(matched.mapping, f)
        nx.write_graphml(matched.pred_graph.graph, os.path.join(out_ds, OUT_FILE.MATCHED_SOL.value))
        nx.write_graphml(matched.gt_graph.graph, os.path.join(out_ds, OUT_FILE.MATCHED_GT.value))

    def write_solved(self, tracked, start):
        # TODO: formalize/document that we're writing out and potentially overwriting path
        out_ds = self.get_path_to_out_dir()
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
    
    def get_path_to_out_dir(self):
        return os.path.join(self.data_config.out_root_path, self.data_config.dataset_name)

    def get_path_to_out_file(self, file_key: OUT_FILE):
        out_ds = self.get_path_to_out_dir()
        return os.path.join(out_ds, file_key.value)
    
    def load_solution_for_metrics(self, tracked_detections=None, tracked_edges=None):
        out_ds = self.get_path_to_out_dir()
        if not os.path.exists(out_ds) and (tracked_detections is None or tracked_edges is None):
            raise ValueError(f"Cannot evaluate experiment as detections and edges were not passed,\n" +
                             "and no results exist at {out_ds}. \n" + 
                             "Have you run this experiment?"
                             )
        if tracked_detections is None:
            # load tracked edges and tracked detections
            tracked_detections = pd.read_csv(self.get_path_to_out_file(OUT_FILE.TRACKED_DETECTIONS))
            tracked_edges = pd.read_csv(self.get_path_to_out_file(OUT_FILE.TRACKED_EDGES))
        seg_ims = load_tiff_frames(self.data_config.segmentation_path)
        sol_graph = nx.from_pandas_edgelist(tracked_edges, "u", "v", ["flow", "cost"], create_using=nx.DiGraph)
        det_keys = [self.data_config.frame_key] + self.data_config.location_keys + [self.data_config.value_key] + ['enter_exit_cost', 'div_cost']
        sol_graph.add_nodes_from(tracked_detections[det_keys].to_dict(orient='index').items())

        track_graph = TrackingGraph(
            sol_graph, 
            label_key=self.data_config.value_key, 
            location_keys=self.data_config.location_keys, 
            segmentation=seg_ims
        )
        return track_graph



if __name__ == '__main__':
    from utils import get_scale
    DATA_ROOT = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/'
    ds_name = 'Fluo-N2DL-HeLa_02'
    data_config = TraxData(
        dataset_name=ds_name,
        image_path=os.path.join(DATA_ROOT, '02/'),
        segmentation_path=os.path.join(DATA_ROOT, '02_ST/SEG/'),
        detections_path='/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02_ST/SEG/detections.csv',
        out_root_path='/home/ddon0001/PhD/experiments/no_div_constraint/',
        frame_shape=[700, 1100],
        scale=get_scale(ds_name),
    )
    experiment = Traxperiment(data_config=data_config)
    experiment.tracktour_config.div_constraint = True
    
    tracked = experiment.run()
    
    experiment.data_config.ground_truth_path = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02_GT/TRA/'
    experiment.data_config.value_key = 'label'
    experiment.evaluate()