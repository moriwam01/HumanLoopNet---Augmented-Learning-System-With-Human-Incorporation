# Necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import idx2numpy as idx
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder


# Preprocessing method
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


# Flask application setup
app = Flask(__name__)


def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    image = img_to_array(image)
    image = image / 255.0
    return image


input_shape = (28, 28, 3)
num_classes = 10

# Model define
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compiling model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Traiing the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))


# Preprocessing method
def preprocess_data(data):
    X = []
    y = []

    for index, row in data.iterrows():
        image = load_img(
            r"C:\Users\mariw\Downloads\HumanLoopNet---Augmented-Learning-System-With-Human-Incorporation\Fashion_data\train-images-idx3-ubyte",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
        )
        image = img_to_array(image)
        image = image / 255.0
        X.append(image)
        y.append(row["class_label"])

    X = np.array(X)
    y = np.array(y)

    # Target labels encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y


# Function to retrieve model predictions and images
def get_model_predictions(image_paths):
    predictions = []
    images = []

    for img_path in image_paths:
        image = load_and_preprocess_image(img_path)
        prediction = model.predict(image)
        predictions.append(prediction)
        images.append(img_path)

    return predictions, images


# route to display the model's predictions
@app.route("/display_predictions", methods=["GET"])
def display_predictions():
    predictions, images = get_model_predictions()
    return render_template(
        "display_predictions.html", predictions=predictions, images=images
    )


# route to render the feedback form
@app.route("/get_feedback", methods=["GET"])
def get_feedback():
    return render_template("collect_feedback.html")


# route to collect expert feedback
@app.route("/collect_feedback", methods=["POST"])
def collect_feedback():
    expert_feedback = request.form["feedback"]
    # Stores the expert feedback along with the corresponding data for future use
    with open("expert_feedback.txt", "a") as f:
        f.write(expert_feedback + "\n")  # Appending the feedback to a file
    return "Thank you for your feedback!"


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
