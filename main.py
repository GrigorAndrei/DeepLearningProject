import subprocess

source_path = "D:\\BAMI\\DeepLearning\\DeepLearningProject\\application\\"

# Define the file names in the order you want to run
files = ["env_setup.py", "normalize_data.py", "train_model.py"]
files_path = []

for file in files:
    files_path.append(source_path + file)

# Loop through and run each Python file
for file in files_path:
    try:
        subprocess.run(["python", file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")