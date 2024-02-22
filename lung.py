import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


ROOT_PATH = os.path.dirname(__file__)
# Load the dataset


def load_data():
    dataset = os.path.join(ROOT_PATH, "data", "lungcancer.csv")
    df = pd.read_csv(dataset)
    df.drop(["index", "Patient Id", "Clubbing of Finger Nails"], axis=1, inplace=True)
    st.write(df.columns.tolist())
    return df


df = load_data()

# Map categorical variables to numerical representations
gender_mapping = {"Male": 0, "Female": 1}
alcohol_use_mapping = {"Low": 0, "Moderate": 1, "High": 2}
air_pollution_mapping = {"Low": 0, "Moderate": 1, "High": 2}
occupational_hazards_mapping = {"Low": 0, "Moderate": 1, "High": 2}
genetic_risk_mapping = {"Low": 0, "Moderate": 1, "High": 2}
chronic_lung_disease_mapping = {"None": 0, "Mild": 1, "Severe": 2}
dust_allergy_mapping = {"None": 0, "Mild": 1, "Severe": 2}
balanced_diet_mapping = {"Poor": 0, "Average": 1, "Good": 2}
obesity_mapping = {"Not Obese": 0, "Obese": 1}
smoking_mapping = {"Non-Smoker": 0, "Ex-Smoker": 1, "Current Smoker": 2}
passive_smoker_mapping = {"Low": 0, "Moderate": 1, "High": 2}
chest_pain_mapping = {"Mild": 0, "High": 1, "Severe": 2}
coughing_of_blood_mapping = {"Mild": 0, "Moderate": 1, "Severe": 2}
fatigue_mapping = {"None": 0, "Mild": 1, "Severe": 2}
weight_loss_mapping = {"None": 0, "Mild": 1, "Severe": 2}
shortness_of_breath_mapping = {"None": 0, "Mild": 1, "Severe": 2}
wheezing_mapping = {"None": 0, "Mild": 1, "Severe": 2}
swallowing_difficulty_mapping = {"None": 0, "Mild": 1, "Severe": 2}
frequent_colds_mapping = {"None": 0, "Mild": 1, "Severe": 2}
dry_cough_mapping = {"None": 0, "Mild": 1, "Severe": 2}
snoring_mapping = {"None": 0, "Mild": 1, "Severe": 2}


# Sidebar for user inputs
st.sidebar.title("Lung Cancer Prediction")
st.sidebar.write("Please select your inputs:")


def map_user_input(user_input, mapping_dict):
    return mapping_dict[user_input]


# User inputs
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=50)

# List of features
features = [
    "Gender",
    "Air Pollution",
    "Alcohol Use",
    "Occupational Hazards",
    "Dust Allergy",
    "Genetic Risk",
    "Chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Frequent Colds",
    "Dry Cough",
    "Snoring",
]

# List of options for each feature
options = [
    ["Male", "Female"],
    ["Low", "Moderate", "High"],
    ["Low", "Moderate", "High"],
    ["Low", "Moderate", "High"],
    ["None", "Mild", "Severe"],
    ["Low", "Moderate", "High"],
    ["None", "Mild", "Severe"],
    ["Poor", "Average", "Good"],
    ["Not Obese", "Obese"],
    ["Non-Smoker", "Ex-Smoker", "Current Smoker"],
    ["Low", "Moderate", "High"],
    ["Mild", "High", "Severe"],
    ["Mild", "Moderate", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
    ["None", "Mild", "Severe"],
]

# List of mapping dictionaries
mappings = [
    gender_mapping,
    air_pollution_mapping,
    alcohol_use_mapping,
    occupational_hazards_mapping,
    dust_allergy_mapping,
    genetic_risk_mapping,
    chronic_lung_disease_mapping,
    balanced_diet_mapping,
    obesity_mapping,
    smoking_mapping,
    passive_smoker_mapping,
    chest_pain_mapping,
    coughing_of_blood_mapping,
    fatigue_mapping,
    weight_loss_mapping,
    shortness_of_breath_mapping,
    wheezing_mapping,
    swallowing_difficulty_mapping,
    frequent_colds_mapping,
    dry_cough_mapping,
    snoring_mapping,
]

# Dictionary to store numerical representations
numerical_inputs = {}

# Loop over features
for feature, option, mapping in zip(features, options, mappings):
    user_input = st.sidebar.selectbox(feature, option)
    numerical_inputs[feature] = map_user_input(user_input, mapping)


# Data preprocessing
# Encode categorical variables

df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df = pd.get_dummies(
    df,
    columns=[
        "Alcohol use",
        "Dust Allergy",
        "Balanced Diet",
        "Air Pollution",
        "OccuPational Hazards",
        "Genetic Risk",
        "Obesity",
        "Smoking",
        "chronic Lung Disease",
        "Passive Smoker",
        "Chest Pain",
        "Coughing of Blood",
        "Fatigue",
        "Weight Loss",
        "Shortness of Breath",
        "Wheezing",
        "Swallowing Difficulty",
        "Frequent Cold",
        "Dry Cough",
        "Snoring",
    ],
)

# Split data into features and target variable
X = df.drop("Level", axis=1)
y = df["Level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=98
)

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=98)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": feature_importances}
)

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

# Print feature importances
st.write(feature_importance_df)

# Select the top-k features (e.g., top 10 features)
top_k_features = feature_importance_df.head(10)["Feature"].tolist()

# Filter columns in the dataset to include only the top-k features
X_train_selected = X_train[top_k_features]
X_test_selected = X_test[top_k_features]

# Choose a threshold or determine the number of top features to select
threshold = 0.01  # Example threshold value

# Select features based on the threshold
selected_features = feature_importance_df[
    feature_importance_df["Importance"] >= threshold
]["Feature"].tolist()

# Filter columns in the dataset to include only the selected features
X_train_selected = pd.DataFrame(X_train, columns=selected_features)
X_test_selected = pd.DataFrame(X_test, columns=selected_features)

# Train the Random Forest classifier with the selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Evaluate model performance
y_pred_selected = rf_model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features:", accuracy_selected)

## Concatenate user input with zeros for other features
user_input = [age] + [numerical_inputs[feature] for feature in features[1:]]

print(user_input)
print(numerical_inputs)

other_features = [0] * (X_train.shape[1] - len(user_input))
input_data = np.array([user_input + other_features])


# Define function to predict lung cancer
def predict_lung_cancer(input_data):
    prediction = rf_model.predict(input_data)
    return prediction


st.title("Model Performance")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

if st.button("Predict"):
    prediction = predict_lung_cancer(input_data)
    st.title("Lung Cancer Prediction Result")
    st.write("Predicted Lung Cancer:", "Yes" if prediction[0] == 1 else "No")
