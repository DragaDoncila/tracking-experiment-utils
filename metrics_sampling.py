import json
import networkx as nx
from tracktour import load_tiff_frames
import numpy as np
from traccuracy 

def sample_trajectory_ends(sol):
    terms = np.asarray([node for node in sol.nodes if sol.out_degree(node) == 0], dtype=np.uint64)
    choices = np.random.permutation(terms)
    return choices


if __name__ == '__main__':
    sol_path = '/home/ddon0001/PhD/experiments/scaled/no_merges_all/Fluo-N2DL-HeLa_01/'
    # gt_path = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/Fluo-N2DL-HeLa/01_GT/TRA/'

    sol = nx.read_graphml(sol_path + 'matched_solution.graphml', node_type=int)
    gt_sol = nx.read_graphml(sol_path + 'matched_gt.graphml')
    nx.set_edge_attributes(sol, False, 'sampled')
    nx.set_edge_attributes(gt_sol, False, 'sampled')

    with open(sol_path + 'matching.json', 'r') as f:
        matching = json.load(f)
    gt_to_sol = {item[0]: item[1] for item in matching}
    sol_to_gt = {item[1]: item[0] for item in matching}

    # TODO: how many traj to sample? What's the minimum across all datasets?
    traj_choices = sample_trajectory_ends(sol, n_samples=30)
    
    # solution source and dest ID of edge (if not FN edge)
    src_id = []
    dst_id = []

    # gt source and dest ID of edge (if not FP edge)
    gt_src_id = []
    gt_dst_id = []

    # is this edge a TP, FP or FN edge?
    # FN edges are have their gt_src_id/gt_dst_id stored
    is_tp_edge = []
    is_fp_edge = []
    is_fn_edge = []

    # whether this edge connects parents to children
    is_branching_edge = []

    #TODO: need to make sure we're not repeating edges
    for terminating_node in traj_choices:

        gt_node = sol_to_gt[terminating_node]
        # this is a false termination
        # follow the edge forwards one frame
        if gt_sol.out_degree(gt_node) != 0:
            # check its true successors in GT
            true_dests = list(gt_sol.successors(gt_node))
            for true_dest in true_dests:
                if not gt_sol.edges[gt_node, true_dest]['sampled']:
                    src_id.append(terminating_node)
                    gt_src_id.append(gt_node)
                    gt_dst_id.append(true_dest)
                    is_fn_edge.append(True)
                    is_tp_edge.append(False)
                    is_fp_edge.append(False)
                    # edge may end in FN node
                    if true_dest in gt_to_sol:
                        dst_id.append(gt_to_sol[true_dest])
                    else:
                        dst_id.append(-1)
                    # edge may be a child connection
                    if len(true_dests) > 1:
                        is_branching_edge.append(True)
                    else:
                        is_branching_edge.append(False)
        
        # now that we've dealt with destination node
        # we start traversing the edges backwards
        current_dest = terminating_node
        while len(current_preds := list(sol.predecessors(current_dest))) == 1:
            current_src = current_preds[0]
            if sol.edges[current_src, current_dest]['sampled']:
                continue

            src_id.append(current_src)
            dst_id.append(current_dest)

            current_gt_src = sol_to_gt[current_src]
            current_gt_dst = sol_to_gt[current_dest]
            gt_src_id.append(current_gt_src)
            gt_dst_id.append(current_gt_dst)

            is_fn_edge.append(False)
            # is this edge FP or TP (can't be FN because it's in solution)
            edge_info = sol.edges[current_src, current_dest]
            if edge_info['EdgeFlag.FALSE_POS']:
                is_fp_edge.append(True)
                is_tp_edge.append(False)
            else:
                is_fp_edge.append(False)
                is_tp_edge.append(True)
            
            # is src branching? this may be a TP or FP
            # division
            if sol.out_degree(current_src) > 1:
                is_branching_edge.append(True)
                other_succ = [succ for succ in sol.successors(current_src) if succ != current_dest][0]
                
                if not sol.edges[current_src, other_succ]['sampled']:
                    src_id.append(current_src)
                    dst_id.append(other_succ)
                    gt_src_id.append(current_gt_src)
                    gt_dst_id.append(sol_to_gt[other_succ])
                    is_branching_edge.append(True)
                    
                    edge_info = sol.edges[current_src, other_succ]
                    if edge_info['EdgeFlag.FALSE_POS']:
                        is_fp_edge.append(True)
                        is_tp_edge.append(False)
                    else:
                        is_fp_edge.append(False)
                        is_tp_edge.append(True)
                    is_fn_edge.append(False)
            else:
                is_branching_edge.append(False)
                # this edge is not branching - should it be?
                # check other successors of current_gt_src
                gt_succs = list(gt_sol.successors(current_gt_src))
                if len(gt_succs) > 1:
                    for gt_succ in gt_succs:
                        # make sure we're not processing
                        # current edge again and that we haven't already annotated this edge
                        if gt_succ != current_gt_dst and not gt_sol.edges[current_gt_src, gt_succ]['sampled']:
                            src_id.append(current_src)
                            gt_src_id.append(current_gt_src)
                            gt_dst_id.append(gt_succ)
                            is_branching_edge.append(True)
                            if gt_succ in gt_to_sol:
                                dst_id.append(gt_to_sol[gt_succ])
                            else:
                                dst_id.append(-1)
                            # current_src isn't branching, but current_gt_src is
                            # gt_succ is not the current edge, so this must be an FN edge
                            is_fn_edge.append(True)
                            is_fp_edge.append(False)
                            is_tp_edge.append(False)

            # source now becomes destination before
            # next iteration of the loop
            current_dest = current_preds[0]
        
        # we've reached the end of this trajectory
        # is this an FA?
        assert len(current_preds) == 0
        # current_dest is now the appearing node
        # does its gt counterpart have a successor
        app = current_dest
        gt_app = sol_to_gt[app]
        if len(preds := list(gt_sol.predecessors(gt_app))) != 0:
            # this is a false appearance
            # gt_app has a predecessor
            gt_pred = preds[0]
            if gt_pred in gt_to_sol:
                sol_pred = gt_to_sol[gt_pred]
            else:
                sol_pred -1
            if not gt_sol.edges[gt_pred, gt_app]['sampled']:
                src_id.append(sol_pred)
                dst_id.append(app)
                gt_dst_id.append(gt_app)
                is_fn_edge.append(True)
                is_fp_edge.append(False)
                is_tp_edge.append(False)

                pred_succs = list(gt_sol.successors(gt_pred))
                if len(pred_succs) == 1:
                    is_branching_edge.append(False)
                else:
                    # pred_succ is branching. this is a branching edge
                    # and there's an FN edge to the other successor
                    is_branching_edge.append(True)
                    other_succ = [succ for succ in pred_succs if succ != gt_app][0]
                    if not gt_sol.edges[gt_pred, other_succ]['sampled']:
                        src_id.append(sol_pred)
                        if other_succ in gt_to_sol:
                            dst_id.append(gt_to_sol[other_succ])
                        else:
                            dst_id.append(-1)
                        gt_src_id.append(gt_pred)
                        gt_dst_id.append(other_succ)
                        is_fn_edge.append(True)
                        is_fp_edge.append(False)
                        is_tp_edge.append(False)
                        is_branching_edge.append(True)



        
