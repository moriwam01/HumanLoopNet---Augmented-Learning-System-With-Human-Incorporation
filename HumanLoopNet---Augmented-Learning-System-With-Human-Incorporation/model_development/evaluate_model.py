# evaluate_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import idx2numpy as idx
import tensorflow as tf

# Parameters adjusting
history = model.fit(
    X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val)
)

X_test = X_test.reshape(-1, 28, 28, 1)
X_test = X_test.astype("float32")
X_test /= 255

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print("\nTest accuracy:", test_acc)

# Define the output file path
output_file_path = "output.txt"

# Check if the provided folder exists and is not empty
if not os.path.exists(directory_path):
    print(f"Error: The directory path '{directory_path}' does not exist.")
    sys.exit(1)
if not os.listdir(directory_path):
    print(f"Error: The directory path '{directory_path}' is empty.")
    sys.exit(1)
else:
    # Evaluate the model
    with open(output_file_path, "w") as f:
        with redirect_stdout(f):
            # Write the model's architecture summary
            print("Model's architecture summary:")
            model.summary()

            # Evaluation metrics
            print(f"\nEvaluation metric(s) obtained: {test_acc}")

            # Additional insights or observations
            print("\nAdditional insights or observations: ")

    print("Evaluation completed. Check the output.txt file for the results.")

# Clean exit
sys.exit(0)
