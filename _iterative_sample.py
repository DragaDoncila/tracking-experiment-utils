"""
In this module we explore sampling strategies for iterative learning.

Our goal is to find the sampling strategy that allows the classifier to
approach its prediction accuracy when trained on the entire dataset with 
the minimal number of samples.
"""

# load initial solution stuff
# compute migraion features
# train classifier using train/test split on solution edges
# store accuracy on test set
# predict on all edges, store pred. accuracy

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from _iterative_solve import load_sol_files, assign_migration_features

def populate_target_label(all_edges, solution_graph, ground_truth, sol_to_gt):
    # assign True to
    # - edges present in the solution and not labelled FP
    # - edges NOT present in the solution, but present in the GT
    # NOTE: this is STRONGER than our "oracle" because we don't ask user about
    # edges not present in solution
    # def is_edge_in_correct_solution(edge):
    #     u, v = edge['u'], edge['v']
    #     edge_in_solution = solution_graph.has_edge(u, v) 
    #     edge_in_solution_correct = edge_in_solution and not solution_graph.edges[u, v]['EdgeFlag.FALSE_POS']
    #     edge_in_gt = False
    #     if u in sol_to_gt and v in sol_to_gt:
    #         gt_u = sol_to_gt[u]
    #         gt_v = sol_to_gt[v]
    #         edge_in_gt = ground_truth.has_edge(gt_u, gt_v) and ground_truth.edges[gt_u, gt_v]['EdgeFlag.FALSE_NEG']
    #     return edge_in_solution_correct or edge_in_gt
    

    def is_edge_in_gt(edge):
        u, v = edge['u'], edge['v']
        # these two nodes exist in the ground truth
        # note this should be all nodes because we have no FPs
        if u in sol_to_gt and v in sol_to_gt:
            gt_u = sol_to_gt[u]
            gt_v = sol_to_gt[v]
            # if this edge is in GT, we mark it as correct
            return ground_truth.has_edge(gt_u, gt_v)
        # if real vertices aren't in GT we have a problem because we have no FPs
        if u >= 0 and v >= 0:
            raise ValueError(f'Node {u} in GT: {u in sol_to_gt}, Node {v} in GT: {v in sol_to_gt}')
        return False
      
    target_labels = all_edges[['u', 'v']].apply(is_edge_in_gt, axis=1)
    all_edges['oracle_is_correct'] = target_labels

def get_baseline_accuracy(all_edges, ft_names):
    all_edges['classifier_pred'] = False
    all_edges['classifier_prob_correct'] = -1.0
    all_edges['test_set'] = False
    real_edge_mask = (all_edges.u >= 0) & (all_edges.v >= 0)
    real_edge_df = all_edges[real_edge_mask][ft_names + ['oracle_is_correct']]
    X = real_edge_df.drop(columns=['oracle_is_correct'])
    y = real_edge_df['oracle_is_correct']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except ValueError:
        return 0, 0
    

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=250, max_depth=10))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    all_edges.loc[X.index, 'classifier_pred'] = clf.predict(X)
    predict_proba = clf.predict_proba(X)

    # clf = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced'))
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # all_edges.loc[X.index, 'classifier_pred'] = clf.predict(X)
    # predict_proba = clf.predict_proba(X)

    sol_edges = all_edges[(all_edges.flow > 0) & real_edge_mask][ft_names]
    sol_y = all_edges[(all_edges.flow > 0) & real_edge_mask]['oracle_is_correct']
    sol_pred = clf.predict(sol_edges)

    model_accuracy = accuracy_score(y_test, y_pred)
    if predict_proba.shape[1] > 1:
        all_edges.loc[X.index, 'classifier_prob_correct'] = predict_proba[:, 1]
    else:
        all_edges.loc[X.index, 'classifier_prob_correct'] = predict_proba
    all_edges.loc[X_test.index, 'test_set'] = True
    
    sol_accuracy = accuracy_score(sol_y, sol_pred)
    return sol_accuracy, model_accuracy

if __name__ == '__main__':
    root_pth = '/home/ddon0001/PhD/experiments/misc/no_merges_small_k/'
    ds_summary_path = os.path.join(root_pth, 'summary.csv')
    ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'det_path']]

    save_root = '/home/ddon0001/PhD/experiments/sampling_pred_no_merge_small_k/'

    ds_names = []
    sol_accuracies = []
    model_accuracies = []
    for i, row in tqdm(ds_info.iterrows()):
        ds_name = row['ds_name']
        out_pth = os.path.join(save_root, f'{ds_name}.csv')
        if os.path.exists(out_pth):
            print(f'{ds_name} already exists')
            continue
        det_pth = row['det_path']
        sol_pth = os.path.join(root_pth, ds_name)
        all_vertices, all_edges, solution_graph, gt_graph, gt_to_sol, sol_to_gt = load_sol_files(sol_pth)
        det_df = pd.read_csv(det_pth)
        ft_names = assign_migration_features(all_edges, det_df)
        populate_target_label(all_edges, solution_graph, gt_graph, sol_to_gt)
        try:
            sol_accuracy, model_accuracy = get_baseline_accuracy(all_edges, ft_names)
        except ValueError as e:
            print('Skipping', ds_name, ':', str(e))
            continue
        df_of_interest = all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)][['u', 'v', 'flow', 'oracle_is_correct', 'classifier_pred', 'classifier_prob_correct', 'test_set'] + ft_names]
        df_of_interest.to_csv(os.path.join(save_root, f'{ds_name}.csv'))
        ds_names.append(ds_name)
        sol_accuracies.append(sol_accuracy)
        model_accuracies.append(model_accuracy)
    acc_df = pd.DataFrame({'ds_name': ds_names, 'sol_accuracy': sol_accuracies, 'model_accuracy': model_accuracies})
    print(acc_df)

# what's the accuracy on INCORRECT EDGES WITH FLOW >0