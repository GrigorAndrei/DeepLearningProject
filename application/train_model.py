import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

if tf.test.is_gpu_available():
    # GPU is available, use it
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and will be used.")
else:
    print("No GPU available, using CPU.")

# Load your dataset
data = pd.read_csv(".\dataset\data_normalized.csv")
number_of_passes = 10
validation_ratio = 0.3
shuffled_data = shuffle(data)

label_encoder = LabelEncoder()
shuffled_data['Platform'] = label_encoder.fit_transform(shuffled_data['Platform'])
shuffled_data['Genre'] = label_encoder.fit_transform(shuffled_data['Genre'])
shuffled_data['Publisher'] = label_encoder.fit_transform(shuffled_data['Publisher'])
shuffled_data['Developer'] = label_encoder.fit_transform(shuffled_data['Developer'])
shuffled_data['Rating'] = label_encoder.fit_transform(shuffled_data['Rating'])

# Data preprocessing: Prepare X and y
# Excludes the fields that are irrelevant to training the model
X = shuffled_data.drop(['Global_Sales', 'Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], axis=1)
# Sets the target of the models
Y = shuffled_data['Global_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_ratio, random_state=42)

# Define a dictionary to store models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR(),
    "Neural Network": MLPRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(),
    "LightGBM Regressor": LGBMRegressor()
}

best_model = None
best_mse = float("inf")

model_accuracies = {model_name: [] for model_name in models}  # Dictionary to store model accuracies for each pass

# Train and evaluate each model
for model_name, model in models.items():
    
    for epoch in range(number_of_passes):
        print(f"Currently training model {model_name} on epoch number {epoch+1}")
        model_instance = model.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        model_accuracies[model_name].append(mse)  # Save the MSE as accuracy for the current pass

        if mse < best_mse:
            best_mse = mse
            best_model = model_name


# Plot the development of accuracy over 10 passes
# Plot the development of accuracy over 10 passes
plt.figure(figsize=(15, 6))
pass_numbers = list(range(number_of_passes))
for model_name, accuracies in model_accuracies.items():
    plt.plot(pass_numbers, accuracies, label=model_name)

plt.xlabel('Pass Number')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title(f'Development of Model Accuracy Over {number_of_passes} Epochs')

plt.savefig('.\\results\\accuracy_plot.png')  # Save accuracy plot as PNG

# Find the best model
best_mse = {model_name: min(accuracies) for model_name, accuracies in model_accuracies.items()}
best_model = min(best_mse, key=best_mse.get)
print(f"The best model is {best_model} with the lowest MSE: {best_mse[best_model]:.2f}")

# Save the best model to a file
best_model_instance = models[best_model]
joblib.dump(best_model_instance, f".\\results\\{best_model}.pkl")

# Plot the error graph
model_errors = {model_name: [mean_squared_error(y_test, model.predict(X_test))] for model_name, model in models.items()}

# Extract model names and errors as lists
model_names = list(model_errors.keys())
model_mses = [model_errors[model_name][0] for model_name in model_names]

# Create a new figure for the error graph
plt.figure(figsize=(15, 6))
plt.barh(model_names, model_mses, color='skyblue')
plt.xlabel('Mean Squared Error (MSE)')
plt.title('Model Errors')

# Save the graphs as JPEG or PNG files
plt.savefig('.\\results\\error_plot.png')     # Save error plot as PNG

# Save the best model to a file
best_model_instance = models[best_model]
joblib.dump(best_model_instance, "best_model.pkl")
