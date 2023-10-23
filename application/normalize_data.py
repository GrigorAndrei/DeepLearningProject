import shutil
import pandas as pd
import os

def normalize_data():
    file_path = "D:\BAMI\DeepLearning\DeepLearningProject\dataset\Video_Games_Sales_as_at_22_Dec_2016.csv"
    normalized_path = "D:\BAMI\DeepLearning\DeepLearningProject\dataset\data_normalized.csv"

    if os.path.exists(normalized_path):
        # Delete the file
        os.remove(normalized_path)
        print(f"Normalized data has been cleared.")
        
    # Load your dataset
    data = pd.read_csv(file_path)

    # Define the columns you want to check for missing values
    columns_to_check = ['Name','Platform','Year_of_Release',
                        'Genre','Publisher','NA_Sales','EU_Sales',
                        'JP_Sales','Other_Sales','Global_Sales',
                        'Critic_Score','Critic_Count','User_Score',
                        'User_Count','Developer','Rating']

    # Drop rows with missing values in any of the specified columns
    data = data.dropna(subset=columns_to_check)

    # Save the normalized dataset to a new CSV file
    data.to_csv(normalized_path, index=False)


def clear_results():
    folder_path = "D:\\BAMI\\DeepLearning\\DeepLearningProject\\results"
    if os.path.exists(folder_path):
        print(f"Results folder already exist.")
        print(f"Clearing previous results.")
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                # Remove file
                os.unlink(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                # Remove directory and its contents
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")

        print(f"Previous results have been cleared.")
        return
    else:
        os.mkdir(folder_path)

normalize_data()
clear_results()