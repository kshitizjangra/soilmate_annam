"""

Author: Annam.ai IIT Ropar
Team Name: SoilMate
Team Members: Kshitiz Jangra, Harshal Chaudhari
Leaderboard Rank: 62

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

def load_image(image_id, path):
    img = load_img(os.path.join(path, image_id), target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

def load_image_preprocessed(image_id, path):
    img = load_img(os.path.join(path, image_id), target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def prepare_datasets(train_dir, train_labels_df):
    X_raw = np.array([load_image(img_id, train_dir) for img_id in train_labels_df['image_id']])
    X_cls = np.array([load_image_preprocessed(img_id, train_dir) for img_id in train_labels_df['image_id']])
    y = np.ones(len(X_cls))
    return X_raw, X_cls, y

