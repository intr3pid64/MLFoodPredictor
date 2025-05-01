#!/usr/bin/env python3

import sys
import json
import numpy as np
import pandas as pd

def predict_all(csv_path):
    """
    Reads 'model_params.json' (saved by your train script),
    loads the new data from csv_path (cleaned format),
    and returns a list of predictions (strings) via soft-voting
    across logistic regression, random forest, gradient boosting.
    """
    #################################
    # 1. LOAD MODEL PARAMS
    #################################
    with open("model_params.json", "r") as f:
        params = json.load(f)
    
    lr_params = params["lr_params"]       # {coeff, intercept, classes}
    rf_params = params["rf_params"]       # {n_classes, n_features, trees:[...]}
    gb_params = params["gb_params"]       # {n_estimators, learning_rate, init_score, trees:[...], loss}
    label_mapping = params["label_mapping"]
    train_cols = params["train_cols"]
    imputer_stats = params["imputer_stats"]  # median for each column
    
    #################################
    # 2. LOAD & PREP NEW DATA
    #################################
    df_new = pd.read_csv(csv_path)
    
    # remove unneeded columns if present
    for c in ["id","label","movie_raw","drink_raw"]:
        if c in df_new.columns:
            df_new.drop(columns=[c], inplace=True)
    
    # If 'movie_std' or 'drink_std' exist in textual form, do get_dummies
    # (If your "cleaned_data_advanced.csv" is already numeric, maybe you skip this.)
    if "movie_std" in df_new.columns or "drink_std" in df_new.columns:
        df_new = pd.get_dummies(df_new, columns=["movie_std","drink_std"], drop_first=False)
    
    # add missing columns
    for col in train_cols:
        if col not in df_new.columns:
            df_new[col] = 0
    # drop extra columns
    extra_cols = [c for c in df_new.columns if c not in train_cols]
    if extra_cols:
        df_new.drop(columns=extra_cols, inplace=True)
    
    # reorder
    df_new = df_new[train_cols]
    
    # convert to numpy
    X_new = df_new.to_numpy(dtype=float)
    
    #################################
    # 3. IMPUTE MISSING VALUES
    #################################
    # fill any NaNs with the stored medians
    for i, med_val in enumerate(imputer_stats):
        col_data = X_new[:, i]
        nan_mask = np.isnan(col_data)
        col_data[nan_mask] = med_val
    
    n_samples = X_new.shape[0]
    
    #################################
    # 4A. LOGISTIC REGRESSION INFERENCE
    #################################
    lr_coeff   = np.array(lr_params["coeff"])      # shape [C, F]
    lr_inter   = np.array(lr_params["intercept"])  # shape [C]
    # compute logits => X_new dot lr_coeff.T + lr_inter
    logits = X_new @ lr_coeff.T + lr_inter
    # softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    lr_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # shape [N, C]

    #################################
    # 4B. RANDOM FOREST
    #################################
    n_classes_rf = rf_params["n_classes"]
    rf_trees = rf_params["trees"]  # list of dicts

    def traverse_tree(sample, feature, threshold, left, right, value):
        """Traverse a single classification tree node-by-node."""
        node = 0
        while True:
            if left[node] == -1 and right[node] == -1:
                # leaf
                return value[node]  # shape [1, n_classes]
            feat_idx = feature[node]
            thres = threshold[node]
            if sample[feat_idx] <= thres:
                node = left[node]
            else:
                node = right[node]

    # accumulate probability distributions across all trees
    rf_prob_sum = np.zeros((n_samples, n_classes_rf), dtype=float)

    for tree_dict in rf_trees:
        feat = np.array(tree_dict["feature"])
        thr  = np.array(tree_dict["threshold"])
        left = np.array(tree_dict["children_left"])
        right= np.array(tree_dict["children_right"])
        val  = np.array(tree_dict["value"])  # shape [num_nodes, n_classes]
        
        for i in range(n_samples):
            leaf_counts = traverse_tree(X_new[i,:], feat, thr, left, right, val)
            leaf_counts = leaf_counts[0]  # from shape [1, n_classes] to [n_classes]
            denom = np.sum(leaf_counts)
            if denom > 0:
                leaf_probs = leaf_counts / denom
            else:
                leaf_probs = np.ones(n_classes_rf) / n_classes_rf
            rf_prob_sum[i,:] += leaf_probs

    # average
    rf_probs = rf_prob_sum / len(rf_trees)

    #################################
    # 4C. GRADIENT BOOSTING
    #################################
    gb_trees = gb_params["trees"]     # shape [n_estimators, n_classes]
    learning_rate = gb_params["learning_rate"]
    n_estimators = gb_params["n_estimators"]
    # figure out how many classes from length of gb_trees[0]
    n_classes_gb = len(gb_trees[0])
    # raw_scores shape [N, n_classes], start at 0
    raw_scores = np.zeros((n_samples, n_classes_gb), dtype=float)

    # node traversal for each tree
    for stage_idx in range(n_estimators):
        for cls_idx in range(n_classes_gb):
            tree_dict = gb_trees[stage_idx][cls_idx]
            feat = np.array(tree_dict["feature"])
            thr  = np.array(tree_dict["threshold"])
            left = np.array(tree_dict["children_left"])
            right= np.array(tree_dict["children_right"])
            val  = np.array(tree_dict["value"])  # shape [num_nodes, 1]
            
            for i in range(n_samples):
                leaf_val = traverse_tree(X_new[i,:], feat, thr, left, right, val)
                # leaf_val shape [1,1], so take leaf_val[0,0]
                gamma = leaf_val[0,0]
                raw_scores[i, cls_idx] += learning_rate * gamma

    # final softmax
    raw_max = np.max(raw_scores, axis=1, keepdims=True)
    exp_raw = np.exp(raw_scores - raw_max)
    gb_probs = exp_raw / np.sum(exp_raw, axis=1, keepdims=True)

    #################################
    # 5. COMBINE VIA SOFT VOTING
    #################################
    combined_probs = (lr_probs + rf_probs + gb_probs) / 3.0
    class_idx = np.argmax(combined_probs, axis=1)

    # map to original label
    preds = [label_mapping[i] for i in class_idx]
    return preds

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pred.py path_to_cleaned_test_csv")
        sys.exit(1)
    
    test_csv = sys.argv[1]
    predictions = predict_all(test_csv)
    for p in predictions:
        print(p)
