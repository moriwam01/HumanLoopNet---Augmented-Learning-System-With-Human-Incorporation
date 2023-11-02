# human_in_the_loop.py

# Placeholder data for model predictions
model_predictions = ["Class1", "Class2", "Class3", "Class1", "Class2"]


# Function for human-in-the-loop mechanism
def human_loop_mechanism(model_predictions):
    while True:
        print("Model Predictions:")
        for idx, prediction in enumerate(model_predictions):
            print(f"Sample {idx+1}: {prediction}")

        sample_num = input("Enter the sample number to correct (0 to exit): ")
        if sample_num == "0":
            break

        if sample_num.isdigit():
            sample_num = int(sample_num)
            if 1 <= sample_num <= len(model_predictions):
                corrected_label = input("Enter the corrected label for this sample: ")
                model_predictions[sample_num - 1] = corrected_label
                print(f"Updated label for sample {sample_num} to {corrected_label}\n")
            else:
                print("Invalid sample number. Please try again.\n")
        else:
            print("Invalid input. Please enter a valid sample number or 0 to exit.\n")


# Function for user interface
def user_interface():
    print("Welcome to the Human-in-the-Loop System\n")
    print("You will now validate or correct the model predictions.\n")
    human_loop_mechanism(model_predictions)
    print("Thank you for using the Human-in-the-Loop System.")


if __name__ == "__main__":
    user_interface()
