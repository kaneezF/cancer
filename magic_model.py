import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT_PATH = os.path.dirname(__file__)

class MagicModel:
    def __init__(self):
        pass


def load_data(self):
    dataset = os.path.join(ROOT_PATH, "data", "lungcancer.csv")
    df = pd.read_csv(dataset)
    df.drop(['index', 'Patient Id', 'Clubbing of Finger Nails'], axis=1, inplace=True)
    return df


df = load_data()

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=[ 'Alcohol use', 'Dust Allergy', 'Balanced Diet','Air Pollution', 'OccuPational Hazards','Genetic Risk','Obesity', 'Smoking','chronic Lung Disease' ,
                                  'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 
                                  'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 
                                   'Frequent Cold', 'Dry Cough','Snoring'])

# Split data into features and target variable
X = df.drop('Level', axis=1)
y = df['Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top-k features (e.g., top 10 features)
top_k_features = feature_importance_df.head(10)['Feature'].tolist()

# Filter columns in the dataset to include only the top-k features
X_train_selected = X_train[top_k_features]
X_test_selected = X_test[top_k_features]

# Choose a threshold or determine the number of top features to select
threshold = 0.01  # Example threshold value

# Select features based on the threshold
selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

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
other_features = [0] * (X_train.shape[1] - len(user_input))

def predict_lung_cancer(input_data):
    prediction = rf_model.predict(input_data)
    return prediction