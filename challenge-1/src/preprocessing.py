"""

Author: Annam.ai IIT Ropar
Team Name: SoilMate
Team Members: Kshitiz Jangra, Harshal Chaudhari
Leaderboard Rank: 62

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

img_size = (224, 224)

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img)
    return img

def load_train_data(train_dir, labels_path):
    # Load labels CSV
    train_labels_df = pd.read_csv(labels_path)
    train_labels_df['soil_type'] = train_labels_df['soil_type'].str.strip()

    # Map soil types to numeric labels
    soil_type_mapping = {soil: idx for idx, soil in enumerate(train_labels_df['soil_type'].unique())}
    train_labels_df['numeric_label'] = train_labels_df['soil_type'].map(soil_type_mapping)

    # Gather image paths and labels
    image_paths, labels = [], []
    for index, row in train_labels_df.iterrows():
        image_path = os.path.join(train_dir, row['image_id'])
        image_paths.append(image_path)
        labels.append(row['numeric_label'])

    labels = to_categorical(labels, num_classes=len(soil_type_mapping))
    image_arrays = np.array([load_and_preprocess_image(p) for p in image_paths])

    return image_arrays, labels, soil_type_mapping, train_labels_df
