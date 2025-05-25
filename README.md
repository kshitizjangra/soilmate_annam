# soilmate_annam
This project is created by Harshal &amp; Kshitiz for the Preliminary round submission for annam.ai 

YouTube Video Link is mentioned below along with the video transcript as well:-
1. Part 1 - Soil Classification: [https://youtu.be/r9Vu5ybXhcs?si=1hxIjuYcU0sKPjJl](url)
2. Part 2 - Soil Classification: [https://youtu.be/C68lbsJp3pY?si=7-TBVu9VBXnBsk0T](url)

Transcript:-
1. Part 1 - Soil Classification: [https://drive.google.com/file/d/16OqIgAKHJAyqkT2gWWe52igxtIfvHebz/view?usp=sharing](url)
2. Part 2 - Soil Classification: [https://drive.google.com/file/d/17xEX2U2S8jRY_WOJxkFMuYjK3bu8mAfq/view?usp=drive_link](url)


✅ Setup & Running Instructions
1. Clone the repository
```
git clone https://github.com/thatharshal/soilmate_annam.git
cd soilmate_annam
```

2. Download and prepare the dataset manually
Go to the competition page and download the dataset:
Kaggle Soil Classification Competition Data
Extract and organize the files as follows:
```
challenge-1/
└── data/
    ├── train/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── test/
    │   ├── image_101.jpg
    │   ├── image_102.jpg
    │   └── ...
    ├── train_labels.csv
    └── test_ids.csv
```
If the data/ folder does not exist, create it manually inside challenge-1/ and place the files accordingly.


3. Install required dependencies
```
pip install -r requirements.txt
```

4. Preprocess the data
```
from src.preprocessing import preprocess_images
preprocess_images(train_dir='challenge-1/data/train', img_size=(224, 224))
```

5. Train the model
Open and run the notebook:
```
notebooks/training.ipynb
```
This notebook will:
Load and preprocess the data

Train a MobileNetV2-based classifier
Save the trained model at challenge-1/working/soil_classification_model.keras

6. Run inference and save predictions
Open and run the notebook:
```
notebooks/inference.ipynb
```
It will:
Load the trained model
Predict on the test dataset
Save predictions to challenge-1/working/final_submission.csv
Generate evaluation metrics at docs/cards/ml-metrics.json
