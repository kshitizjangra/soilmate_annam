"""

Author: Annam.ai IIT Ropar
Team Name: SoilMate
Team Members: Kshitiz Jangra, Harshal Chaudhari
Leaderboard Rank: 62

"""

# Here you add all the post-processing related details for the task completed from Kaggle.

import os
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = (224, 224)

def plot_training_history(history):
    plt.figure(figsize=(16, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val, soil_type_mapping):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Print F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Validation F1-Score: {f1:.4f}")

    # Classification report and confusion matrix
    print("Classification Report:\n", classification_report(y_true, y_pred))
    conf_mat = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=soil_type_mapping.keys(),
                yticklabels=soil_type_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Save metrics to ml-metrics.json
    reverse_mapping = {v: k for k, v in soil_type_mapping.items()}
    class_f1_scores = f1_score(y_true, y_pred, average=None)
    per_class_f1 = {
        reverse_mapping[i]: round(score, 2) for i, score in enumerate(class_f1_scores)
    }

    metrics = {
        "_comment": "This JSON file containing the ml-metrics",
        "Name": "Kshitiz Jangra, Harshal Chaudhari",
        "Kaggle Username": "annam.ai",
        "Team Name": "SoilMate",
        "f1 scores": {
        "_comment": "Here are the class wise f1 scores",
            **per_class_f1
        }
    }

    os.makedirs("docs/cards", exist_ok=True)
    with open("docs/cards/ml-metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def preprocess_and_predict(image_id, model, test_dir, mapping):
    image_path = os.path.join(test_dir, image_id)
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping.get(predicted_label, 'Unknown')
