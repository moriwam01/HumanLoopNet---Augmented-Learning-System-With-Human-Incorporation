import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def analyze_data(directory_path):
    # List of all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    # Initializing an empty list to store DataFrames
    dataframes = []

    # Iterating through the list of CSV files and loading them into DataFrames
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # Concatenating all DataFrames into a single DataFrame
    data = pd.concat(dataframes, axis=0, ignore_index=True)

    return data


def visualize_data(data):
    # Extracting the 'label' column
    class_labels = data["label"]

    # Visualizing the distribution of classes
    plt.figure(figsize=(10, 6))
    sns.countplot(x=class_labels)
    plt.title("Distribution of Classes")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    # Directory path
    directory_path = r"C:\Users\mariw\Downloads\HumanLoopNet---Augmented-Learning-System-With-Human-Incorporation\Fashion_data"

    data = analyze_data(directory_path)
    visualize_data(data)
