import subprocess

# List of required packages
required_packages = [
    'pandas',
    'scikit-learn',
    'numpy',
    'xgboost',
    'lightgbm',
    'joblib',
    'tensorflow',
    'matplotlib',
]

# Install each package using pip
for package in required_packages:
    try:
        subprocess.check_call(['pip', 'install', package])
    except Exception as e:
        print(f"Error installing {package}: {e}")

print("All required packages have been successfully installed.")
