"""

Author: Annam.ai IIT Ropar
Team Name: SoilMate
Team Members: Kshitiz Jangra, Harshal Chaudhari
Leaderboard Rank: 62

"""

# Here you add all the post-processing related details for the task completed from Kaggle.

import numpy as np
import json
from sklearn.metrics import f1_score

def evaluate_combined(mse_val, val_probs, ae_threshold, cls_threshold):
    y_pred_ae = (mse_val <= ae_threshold).astype(int)
    y_pred_cls = (val_probs >= cls_threshold).astype(int)
    y_pred_combined = y_pred_ae & y_pred_cls
    f1_combined = f1_score(np.ones_like(y_pred_combined), y_pred_combined)
    return f1_combined, y_pred_combined

def save_metrics(f1_combined, path='docs/cards/ml-metrics.json'):
    ml_metrics = {
        "_comment": "This JSON file containing the ml-metrics",
        "Name": "Annam.ai",
        "Kaggle Username": "annam.ai",
        "Team Name": "soilclassifiers",
        "f1 scores": {
            "_comment": "Here are the class wise f1 scores",
            "alluvial soil": f1_combined,
            "red soil": f1_combined,
            "black soil": f1_combined,
            "clay soil": f1_combined
        }
    }
    with open(path, 'w') as f:
        json.dump(ml_metrics, f, indent=2)
