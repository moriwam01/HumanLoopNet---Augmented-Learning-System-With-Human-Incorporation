# preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import idx2numpy as idx

# Image size
IMG_WIDTH, IMG_HEIGHT = 100, 100


def preprocess_data(data):
    X = []
    y = []

    for index, row in data.iterrows():
        image = load_img(
            r"C:\Users\mariw\Downloads\HumanLoopNet---Augmented-Learning-System-With-Human-Incorporation\Fashion_data\train-images-idx3-ubyte",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
        )
        image = img_to_array(image)
        image = image / 255.0  # Normalize the pixel values
        X.append(image)
        y.append(row["class_label"])

    X = np.array(X)
    y = np.array(y)

    # Encoding the target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y
