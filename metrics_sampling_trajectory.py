import json
import os
import networkx as nx
import numpy as np
from tqdm import tqdm

def sample_trajectory_ends(sol):
    terms = np.asarray([node for node in sol.nodes if sol.out_degree(node) == 0], dtype=np.uint64)
    choices = np.random.permutation(terms)
    return choices

def check_false_termination(gt_sol, terminating_node, sol_to_gt, sampled_id):
    gt_node = sol_to_gt[terminating_node]
    # this is a false termination
    # follow the edge forwards one frame
    if gt_sol.out_degree(gt_node) != 0:
        # check its true successors in GT
        true_dests = list(gt_sol.successors(gt_node))
        for true_dest in true_dests:
            if not gt_sol.edges[gt_node, true_dest]['sampled']:
                gt_sol.edges[gt_node, true_dest]['sampled'] = True
                gt_sol.edges[gt_node, true_dest]['sampled_id'] = sampled_id
                gt_sol.edges[gt_node, true_dest]['sampled_fn'] = True

                sampled_id += 1
    return sampled_id

def check_false_appearance(gt_sol, app, sol_to_gt, sampled_id):
    gt_app = sol_to_gt[app]
    if len(preds := list(gt_sol.predecessors(gt_app))) != 0:
        # this is a false appearance
        # gt_app has a predecessor
        gt_pred = preds[0]
        if not gt_sol.edges[gt_pred, gt_app]['sampled']:
            gt_sol.edges[gt_pred, gt_app]['sampled'] = True
            gt_sol.edges[gt_pred, gt_app]['sampled_id'] = sampled_id
            sampled_id += 1
            gt_sol.edges[gt_pred, gt_app]['sampled_fn'] = True

            pred_succs = list(gt_sol.successors(gt_pred))
            if len(pred_succs) > 1:
                # pred_succ is branching. this is a branching edge
                # and there's an FN edge to the other successor
                other_succ = [succ for succ in pred_succs if succ != gt_app][0]
                if not gt_sol.edges[gt_pred, other_succ]['sampled']:
                    gt_sol.edges[gt_pred, other_succ]['sampled'] = True
                    gt_sol.edges[gt_pred, other_succ]['sampled_id'] = sampled_id
                    sampled_id += 1
                    gt_sol.edges[gt_pred, other_succ]['sampled_fn'] = True
    return sampled_id


if __name__ == '__main__':
    out_root = '/home/ddon0001/PhD/experiments/sparse_sampling/trajectory/'
    sol_root = '/home/ddon0001/PhD/experiments/scaled/no_merges_all/'
    all_ds_names = [item for item in os.listdir(sol_root) if os.path.isdir(os.path.join(sol_root, item))]
    
    for ds_name in tqdm(all_ds_names):
        sol_path = sol_root + ds_name + '/'

        sol = nx.read_graphml(sol_path + 'matched_solution.graphml', node_type=int)
        gt_sol = nx.read_graphml(sol_path + 'matched_gt.graphml')
        nx.set_edge_attributes(sol, False, 'sampled')
        nx.set_edge_attributes(sol, False, 'sampled_id')
        nx.set_edge_attributes(sol, False, 'sampled_fp')
        nx.set_edge_attributes(sol, False, 'sampled_tp')

        # get edges only get marked as sampled when they're FN
        nx.set_edge_attributes(gt_sol, False, 'sampled')
        nx.set_edge_attributes(gt_sol, False, 'sampled_id')
        nx.set_edge_attributes(gt_sol, False, 'sampled_fn')
        

        with open(sol_path + 'matching.json', 'r') as f:
            matching = json.load(f)
        gt_to_sol = {item[0]: item[1] for item in matching}
        sol_to_gt = {item[1]: item[0] for item in matching}

        # TODO: how many traj to sample? What's the minimum across all datasets?
        traj_choices = sample_trajectory_ends(sol)

        sampled_id = 0
        #TODO: need to make sure we're not repeating edges
        for terminating_node in traj_choices:

            sampled_id = check_false_termination(gt_sol, terminating_node, sol_to_gt, sampled_id)
            
            # now that we've dealt with destination node
            # we start traversing the edges backwards
            current_dest = terminating_node
            while len(current_preds := list(sol.predecessors(current_dest))) == 1:
                current_src = current_preds[0]
                # this edge has been sampled meaning we've already seen this trajectory
                if sol.edges[current_src, current_dest]['sampled']:
                    break
                sol.edges[current_src, current_dest]['sampled'] = True
                sol.edges[current_src, current_dest]['sampled_id'] = sampled_id
                sampled_id += 1
                current_gt_src = sol_to_gt[current_src]
                current_gt_dst = sol_to_gt[current_dest]
                # is this edge FP or TP (can't be FN because it's in solution)
                # we don't use the edge flag because of discrepancies in
                # how the CTC edges are assigned
                if not gt_sol.has_edge(current_gt_src, current_gt_dst):
                    sol.edges[current_src, current_dest]['sampled_fp'] = True
                else:
                    # mark TP because we can derive WS edges from the branching
                    # status of src node and its corresponding gt src node
                    sol.edges[current_src, current_dest]['sampled_tp'] = True
                
                # is src branching? this may be a TP or FP
                # division
                if sol.out_degree(current_src) > 1:
                    # is_branching_edge.append(True)
                    other_succ = [succ for succ in sol.successors(current_src) if succ != current_dest][0]
                    
                    if not sol.edges[current_src, other_succ]['sampled']:
                        sol.edges[current_src, other_succ]['sampled'] = True
                        sol.edges[current_src, other_succ]['sampled_id'] = sampled_id
                        sampled_id += 1
                        gt_src = sol_to_gt[current_src]
                        gt_succ = sol_to_gt[other_succ]
                        if not gt_sol.has_edge(gt_src, gt_succ):
                            sol.edges[current_src, other_succ]['sampled_fp'] = True
                        else:
                            sol.edges[current_src, other_succ]['sampled_tp'] = True

                # it could be that we have an instance of incorrect children
                # so we should check if gt src node is branching
                # but attached to different children

                gt_succs = list(gt_sol.successors(current_gt_src))
                if len(gt_succs) > 1:
                    for gt_succ in gt_succs:
                        # make sure we're not processing
                        # current edge again and that we haven't already annotated this edge
                        if gt_succ != current_gt_dst and not gt_sol.edges[current_gt_src, gt_succ]['sampled']:
                            gt_sol.edges[current_gt_src, gt_succ]['sampled'] = True
                            gt_sol.edges[current_gt_src, gt_succ]['sampled_id'] = sampled_id
                            sampled_id += 1
                            # check if FN edge
                            is_fn = False
                            if gt_succ not in gt_to_sol:
                                is_fn = True
                            else:
                                sol_succ = gt_to_sol[gt_succ]
                                if not sol.has_edge(current_src, sol_succ):
                                    is_fn = True
                            gt_sol.edges[current_gt_src, gt_succ]['sampled_fn'] = is_fn

                # source now becomes destination before
                # next iteration of the loop
                current_dest = current_preds[0]
            
            # we've reached the end of this trajectory
            # it could be an appearance or it could just be a sampled branch
            if len(current_preds) == 0:
                # current_dest is now the appearing node
                # does its gt counterpart have a successor
                app = current_dest
                sampled_id = check_false_appearance(gt_sol, app, sol_to_gt, sampled_id)

        nx.write_graphml(sol, out_root + ds_name + '_sampled.graphml')
        nx.write_graphml(gt_sol, out_root + ds_name + '_gt_sampled.graphml')
        
