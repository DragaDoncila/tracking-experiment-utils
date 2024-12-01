import os
import networkx as nx
import numpy as np
import pandas as pd

from cellpose import models
from scipy.ndimage import distance_transform_edt
from skimage.filters import gaussian
from skimage.filters.rank import mean
from skimage.morphology import disk, ball
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from tifffile import imwrite
from tracktour import load_tiff_frames


def str_to_int_coords_tuple(coord_str):
    coord_list = coord_str.lstrip("[").lstrip(" ").rstrip("]").split(" ")
    coords = []
    for coord_str in coord_list:
        if len(coord_str.strip(" ")):
            coords.append(float(coord_str))
    return tuple([int(x) for x in coords])


def get_coords(node_info):
    return np.asarray(
        [node_info[c] for c in (["z", "y", "x"] if "z" in node_info else ["y", "x"])]
    )


def save_seg(masks, pth):
    os.makedirs(pth, exist_ok=True)
    n_digits = max(len(str(len(masks))), 3)
    for i, frame in enumerate(masks):
        frame_out_name = os.path.join(pth, f"mask{str(i).zfill(n_digits)}.tif")
        imwrite(frame_out_name, frame, compression="zlib")


def resolve_both_overlap(seg, merge_t, merge_label, next_label, child_one, child_two):
    frame_of_interest = seg[merge_t]
    merge_label_mask = frame_of_interest == merge_label
    markers = np.zeros_like(merge_label_mask, dtype=np.uint32)

    markers[child_one] = next_label
    next_label += 1
    markers[child_two] = next_label
    next_label += 1

    marker_bool_mask = markers == 0
    marker_distance = distance_transform_edt(marker_bool_mask)
    new_labels = watershed(marker_distance, markers=markers, mask=merge_label_mask)
    seg[merge_t][merge_label_mask] = new_labels[merge_label_mask]
    return seg, "split", next_label


def get_average_cell_displacements(sol):
    """Get average frame-to-frame cell displacement in solution.

    Displacement is returned in absolute value (directionless)

    Parameters
    ----------
    sol : nx.DiGraph
        solution graph

    Returns
    -------
    np.ndarray
        array of displacement along each axis ([z], y, x)
    """
    edge_source_coords = []
    edge_dest_coords = []
    for u, v in sol.edges:
        u_info = sol.nodes[u]
        v_info = sol.nodes[v]
        edge_source_coords.append(get_coords(u_info))
        edge_dest_coords.append(get_coords(v_info))
    edge_source_coords = np.asarray(edge_source_coords)
    edge_dest_coords = np.asarray(edge_dest_coords)
    edge_displacements = edge_dest_coords - edge_source_coords
    average_displacements = np.abs(np.mean(edge_displacements, axis=0))
    return average_displacements


def get_average_cell_length(seg):
    """Get average length of major axis of cells in segmentation.

    Parameters
    ----------
    seg : np.ndarray
        array of segmentation frames

    Returns
    -------
    float
        average length of major axis across all cells.
    """
    axis_major_lengths = []
    for frame in seg:
        axis_major_lengths.extend(
            list(regionprops_table(frame, properties=["axis_major_length"]).values())[0]
        )
    average_cell_length = np.mean(axis_major_lengths)
    return average_cell_length


def get_bounding_box_displacement(sol, seg):
    """Get displacement from centroid to use for bounding box.

    Returns 2 * average cell displacement in each axis
      + 2 * average cell length

    Parameters
    ----------
    sol : nx.DiGraph
        solution graph
    seg : np.ndarray
        segmentation frames

    Returns
    -------
    np.ndarray
        array of bounding box displacement across each axis
    float
        average cell length
    """
    average_axis_displacements = get_average_cell_displacements(sol)
    average_cell_length = get_average_cell_length(seg)
    return average_axis_displacements * 2 + average_cell_length * 1.5, average_cell_length


def get_bounding_box_slice(
    cell_one_coords, cell_two_coords, bounding_box_displacement, frame_shape
):
    """Get bounding box slice for two children.

    Parameters
    ----------
    child_one : tuple
        coordinates of first child
    child_two : tuple
        coordinates of second child
    bounding_box_displacement : np.ndarray
        array of bounding box displacement across each axis
    frame_shape : tuple
        shape of each frame

    Returns
    -------
    tuple
        bounding box slice for two children
    """
    min_coords = []
    max_coords = []
    for ax in range(len(cell_one_coords)):
        ax_min = min(cell_one_coords[ax], cell_two_coords[ax])
        # the min values get (2*average_cell_length + 2*axis_displacement) subtracted
        ax_min = max(0, ax_min - bounding_box_displacement[ax])
        min_coords.append(int(ax_min))

        # the max values get (2*average_cell_length + 2*axis_displacement) added
        ax_max = max(cell_one_coords[ax], cell_two_coords[ax])
        ax_max = min(frame_shape[ax], ax_max + bounding_box_displacement[ax])
        max_coords.append(int(ax_max))

    bounding_slice = tuple(slice(m, M + 1) for m, M in zip(min_coords, max_coords))
    return bounding_slice

def get_bounding_box_slice_single(child, bounding_box_displacement, frame_shape):
    min_coords = []
    max_coords = []
    for ax in range(len(child)):
        ax_min = max(0, child[ax] - bounding_box_displacement[ax])
        ax_max = min(frame_shape[ax], child[ax] + bounding_box_displacement[ax])
        min_coords.append(int(ax_min))
        max_coords.append(int(ax_max))
    bounding_slice = tuple(slice(m, M + 1) for m, M in zip(min_coords, max_coords))
    return bounding_slice


def resolve_single_overlap(
    seg,
    img,
    merge_t,
    next_label,
    child_one,
    child_two,
    intersecting_child_one,
    bounding_box_displacement,
    average_cell_length,
    seg_model
):
    frame_of_interest_seg = seg[merge_t]
    frame_of_interest_img = img[merge_t]
    new_masks = np.zeros_like(frame_of_interest_seg)
    # bounding_slice = get_bounding_box_slice(child_one, child_two, bounding_box_displacement, frame_of_interest_seg.shape)
    bounding_slice = get_bounding_box_slice_single(child_two if intersecting_child_one else child_one, bounding_box_displacement, frame_of_interest_seg.shape)
    img_region = frame_of_interest_img[bounding_slice]

    do_3d = False
    # footprint = disk(5)
    if len(frame_of_interest_img.shape) == 3:
        do_3d = True
    #     footprint = ball(5)

    # filtered = mean(img_region, footprint=footprint)
    filtered = gaussian(img_region, sigma=0.4)
    masks_pred, _, _, _ = seg_model.eval(filtered, diameter=average_cell_length, channels=[0, 0], do_3D=do_3d)
    
    other_child_coords = child_two if intersecting_child_one else child_one
    new_masks[bounding_slice] = masks_pred
    if (new_label := new_masks[other_child_coords]) != 0: 
        # check that this new label isn't intersecting other cells in frame
        just_label_reg = masks_pred == new_label
        intersect = just_label_reg + frame_of_interest_seg[bounding_slice].astype(bool)
        if np.any(intersect == 2):
            raise ValueError("New label intersects with other cells")
        just_label = new_masks == new_label
        seg[merge_t][just_label] = next_label
        next_label += 1
        return seg, "introduce", next_label

    # TODO: just use image region around other overlapping child?
    # couldn't find anything so we just return
    return seg, 'None', next_label


if __name__ == "__main__":
    sol_root = "/home/ddon0001/PhD/experiments/scaled/no_div_constraint_err_seg"
    data_root = "/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/"
    projected_children_path = (
        "/home/ddon0001/PhD/experiments/merge_resolution/projected_children_all.csv"
    )

    merge_resol_root = "/home/ddon0001/PhD/experiments/merge_resolution/"
    projected_children_out_path = "/home/ddon0001/PhD/experiments/merge_resolution/projected_children_resolved_small_reg_no_split_gaussian.csv"

    projected_children_df = pd.read_csv(projected_children_path)
    # We save the decision we made as a result of the projected children
    projected_children_df["decision"] = "Nothing"
    projected_children_df["new_label_1"] = -1
    projected_children_df["new_label_2"] = -1

    # for each dataset, load it, resolve the both overlap and single overlap
    # cases, re-save the updated segmentation with any newly introduced
    # nodes, and update the projected children with the decision made
    for ds_name in projected_children_df["ds_name"].unique():
        ds, seq = ds_name.split("_")

        # Load solution, image and segmentation
        solution_path = f"{sol_root}/{ds_name}/matched_solution.graphml"
        seg_path = f"{data_root}/{ds}/{seq}_ERR_SEG/"
        img_path = f"{data_root}/{ds}/{seq}/"

        seg_out = f"{merge_resol_root}/{ds}/{seq}_ERR_SEG/"

        sol = nx.read_graphml(solution_path, node_type=int)
        seg = load_tiff_frames(seg_path)
        img = load_tiff_frames(img_path)

        both_overlap = projected_children_df[
            (projected_children_df.child_one_overlap_merge == True)
            & (projected_children_df.child_two_overlap_merge == True)
        ]
        both_overlap_ds = both_overlap[both_overlap.ds_name == ds_name]

        single_overlap = projected_children_df[
            (
                projected_children_df.child_one_overlap_merge
                != projected_children_df.child_two_overlap_merge
            )
        ]
        actual_single_overlap = single_overlap[
            (single_overlap.child_one_overlap_other == False)
            & (single_overlap.child_two_overlap_other == False)
        ]
        single_overlap_ds = actual_single_overlap[
            actual_single_overlap.ds_name == ds_name
        ]

        next_label = np.max(seg) + 1
        # for row in both_overlap_ds.itertuples():
        #     merge_id = row.merge_id
        #     merge_info = sol.nodes[merge_id]
        #     merge_t = merge_info["t"]
        #     merge_label = merge_info["label"]

        #     child_one = str_to_int_coords_tuple(row.projected_child_one_coords)
        #     child_two = str_to_int_coords_tuple(row.projected_child_two_coords)
        #     # TODO: need an area threshold above which we actually keep the split
        #     seg, decision, next_label = resolve_both_overlap(
        #         seg, merge_t, merge_label, next_label, child_one, child_two
        #     )
        #     projected_children_df.loc[row.Index, "decision"] = decision
        #     projected_children_df.loc[row.Index, "new_label_1"] = next_label - 2
        #     projected_children_df.loc[row.Index, "new_label_2"] = next_label - 1

        if len(single_overlap_ds):
            bbox_displacement, average_cell_length = get_bounding_box_displacement(sol, seg)
            seg_model = models.Cellpose(gpu=False, model_type="cyto3")
        for row in single_overlap_ds.itertuples():
            merge_id = row.merge_id
            intersecting_child_one = 0 if row.child_two_overlap_merge else 1
            child_one = str_to_int_coords_tuple(row.projected_child_one_coords)
            child_two = str_to_int_coords_tuple(row.projected_child_two_coords)
            
            merge_info = sol.nodes[merge_id]
            merge_t = merge_info["t"]
            merge_label = merge_info["label"]

            seg, decision, next_label = resolve_single_overlap(
                seg,
                img,
                merge_t,
                next_label,
                child_one,
                child_two,
                intersecting_child_one,
                bbox_displacement,
                average_cell_length,
                seg_model
            )

            # TODO: save decision, save new label
            if decision == 'introduce':
                projected_children_df.loc[row.Index, "decision"] = decision
                projected_children_df.loc[row.Index, "new_label_1"] = next_label - 1

        save_seg(seg, seg_out)
    projected_children_df.to_csv(projected_children_out_path, index=False)