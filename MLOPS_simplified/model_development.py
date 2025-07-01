# A typical script after initial data exploration
# All imports are at the top
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- DATA LOADING ---
# Loading the iris dataset directly
iris_data = load_iris(as_frame=True)
my_dataframe = iris_data.frame
# Renaming columns happens here
my_dataframe.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'target']

# --- FEATURE DEFINITION AND PREPROCESSING ---
# The features are just a list of variables used directly
features_to_use = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
target_variable = 'target'

X = my_dataframe[features_to_use]
y = my_dataframe[target_variable]

# Splitting data right away
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing and fitting the scaler in the main script body
my_scaler = StandardScaler()
X_train_transformed = my_scaler.fit_transform(X_train)

# --- MODEL TRAINING ---
# Model is defined and trained here
the_model = RandomForestClassifier(n_estimators=100, random_state=42)
the_model.fit(X_train_transformed, y_train)
print("Model training complete.")

# --- SAVING ARTIFACTS ---
# Saving the trained model and the scaler for later use
with open('model.pkl', 'wb') as f:
    pickle.dump(the_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(my_scaler, f)
print("Model and scaler have been saved to .pkl files.")

# --- EVALUATION AND INFERENCE LOGIC ---
# The scaler is used on the test set here
X_test_transformed = my_scaler.transform(X_test)
# Predictions are made
preds = the_model.predict(X_test_transformed)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# Example of how inference might be done on a single new data point
print("\n--- Inference Example ---")
# Loading the artifacts back for the example
loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
sample_data = [[5.1, 3.5, 1.4, 0.2]]
sample_data_scaled = loaded_scaler.transform(sample_data)
result = loaded_model.predict(sample_data_scaled)
print(f"Prediction for {sample_data}: {result}")
