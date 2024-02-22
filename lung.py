import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset

def load_data():
    df = pd.read_csv('F:\cancer\lungcancer.csv')
     # Dropping 'index' and 'patient_id' columns
    df.drop(['index', 'Patient Id', 'Clubbing of Finger Nails'], axis=1, inplace=True)
    # Print remaining columns
    st.write(df.columns.tolist())
    return df

df = load_data()

# Map categorical variables to numerical representations
gender_mapping = {'Male': 0, 'Female': 1}
alcohol_use_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
air_pollution_mapping = {'Low':0, 'Moderate':1, 'High':2}
occupational_hazards_mapping = {'Low':0, 'Moderate':1, 'High':2}
genetic_risk_mapping = {'Low':0, 'Moderate':1, 'High':2}
chronic_lung_disease_mapping = {'None':0, 'Mild':1, 'Severe':2}
dust_allergy_mapping = {'None': 0, 'Mild': 1, 'Severe': 2}
balanced_diet_mapping = {'Poor':0 , 'Average':1 , 'Good':2}
obesity_mapping = {'Not Obese':0, 'Obese':1}
smoking_mapping = {'Non-Smoker':0, 'Ex-Smoker':1, 'Current Smoker':2}
passive_smoker_mapping = {'Low':0, 'Moderate':1, 'High':2}
chest_pain_mapping ={'Mild':0, 'High':1, 'Severe':2}
coughing_of_blood_mapping = {'Mild':0, 'Moderate':1, 'Severe':2}
fatigue_mapping = {'None':0, 'Mild':1, 'Severe':2}
weight_loss_mapping = {'None':0, 'Mild':1, 'Severe':2}
shortness_of_breath_mapping = {'None':0, 'Mild':1, 'Severe':2}
wheezing_mapping = {'None':0, 'Mild':1, 'Severe':2}
swallowing_difficulty_mapping = {'None':0, 'Mild':1, 'Severe':2}
frequent_colds_mapping = {'None':0, 'Mild':1, 'Severe':2}
dry_cough_mapping = {'None':0, 'Mild':1, 'Severe':2}
snoring_mapping={'None':0, 'Mild':1, 'Severe':2}


# Sidebar for user inputs
st.sidebar.title("Lung Cancer Prediction")
st.sidebar.write("Please select your inputs:")

# User inputs
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=50)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
air_pollution = st.sidebar.selectbox("Air Pollution", ['Low', 'Moderate', 'High'])
alcohol_use = st.sidebar.selectbox("Alcohol Use", ['Low', 'Moderate', 'High'])
occupational_hazards = st.sidebar.selectbox("OccuPational Hazards", ['Low', 'Moderate', 'High'])
dust_allergy = st.sidebar.selectbox("Dust Allergy", ['None', 'Mild', 'Severe'])
genetic_risk = st.sidebar.selectbox("Genetic Risk", ['Low', 'Moderate', 'High'])
chronic_lung_disease = st.sidebar.selectbox("chronic Lung Disease", ['None', 'Mild', 'Severe'])
balanced_diet = st.sidebar.selectbox("Balanced Diet", ['Poor', 'Average', 'Good'])
obesity = st.sidebar.selectbox("Obesity", ['Not Obese', 'Obese'])
smoking = st.sidebar.selectbox("Smoking", ['Non-Smoker', 'Ex-Smoker', 'Current Smoker'])
passive_smoker = st.sidebar.selectbox("Passive Smoker", ['Low', 'Moderate', 'High'])
chest_pain = st.sidebar.selectbox("Chest Pain", ['Mild', 'High', 'Severe'])
coughing_of_blood = st.sidebar.selectbox("Coughing of Blood", ['Mild', 'Moderate', 'Severe'])
fatigue = st.sidebar.selectbox("Fatigue", ['None', 'Mild', 'Severe'])
weight_loss = st.sidebar.selectbox("Weight Loss", ['None', 'Mild', 'Severe'])
shortness_of_breath = st.sidebar.selectbox("Shortness of Breath", ['None', 'Mild', 'Severe'])
wheezing = st.sidebar.selectbox("Wheezing", ['None', 'Mild', 'Severe'])
swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty", ['None', 'Mild', 'Severe'])
frequent_colds = st.sidebar.selectbox("Frequent Cold", ['None', 'Mild', 'Severe'])
dry_cough = st.sidebar.selectbox("Dry Cough", ['None', 'Mild', 'Severe'])
snoring = st.sidebar.selectbox("Snoring", ['None', 'Mild', 'Severe'])

# Map user inputs to numerical representations
gender_numeric = gender_mapping[gender]
alcohol_use_numeric = alcohol_use_mapping[alcohol_use]
air_pollution_numeric = air_pollution_mapping[air_pollution]
occupational_hazards_numeric = occupational_hazards_mapping[occupational_hazards]
genetic_risk_numeric = genetic_risk_mapping[genetic_risk]
chronic_lung_disease_numeric = chronic_lung_disease_mapping[chronic_lung_disease]
dust_allergy_numeric = dust_allergy_mapping[dust_allergy]
balanced_diet_numeric = balanced_diet_mapping[balanced_diet]
obesity_numeric = obesity_mapping[obesity]
smoking_numeric = smoking_mapping[smoking]
passive_smoker_numeric = passive_smoker_mapping[passive_smoker]
chest_pain_numeric =chest_pain_mapping[chest_pain]
coughing_of_blood_numeric = coughing_of_blood_mapping[coughing_of_blood]
fatigue_numeric = fatigue_mapping[fatigue]
weight_loss_numeric = weight_loss_mapping[weight_loss]
shortness_of_breath_numeric = shortness_of_breath_mapping[shortness_of_breath]
wheezing_numeric = wheezing_mapping[wheezing]
swallowing_difficulty_numeric = swallowing_difficulty_mapping[swallowing_difficulty]
frequent_colds_numeric = frequent_colds_mapping[frequent_colds]
dry_cough_numeric = dry_cough_mapping[dry_cough]
snoring_numeric = snoring_mapping[snoring]


# Data preprocessing
# Encode categorical variables
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

# Print feature importances
st.write(feature_importance_df)

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
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train the Random Forest classifier with the selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Evaluate model performance
y_pred_selected = rf_model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features:", accuracy_selected)

## Concatenate user input with zeros for other features
user_input = [age, alcohol_use_numeric, dust_allergy_numeric,occupational_hazards_numeric,air_pollution_numeric,genetic_risk_numeric,chronic_lung_disease_numeric, balanced_diet_numeric, obesity_numeric, smoking_numeric, passive_smoker_numeric, 
              chest_pain_numeric, coughing_of_blood_numeric, fatigue_numeric, weight_loss_numeric, shortness_of_breath_numeric, wheezing_numeric, 
              swallowing_difficulty_numeric, frequent_colds_numeric, dry_cough_numeric,snoring_numeric]
other_features = [0] * (X_train.shape[1] - len(user_input))
input_data = np.array([user_input + other_features])

# Define function to predict lung cancer
def predict_lung_cancer(input_data):
    prediction = rf_model.predict(input_data)
    return prediction

# Display model performance
st.title("Model Performance")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

if st.button('Predict'):
    # Predict lung cancer
    prediction = predict_lung_cancer(input_data)
    
    # Display prediction
    st.title("Lung Cancer Prediction Result")
    st.write("Predicted Lung Cancer:", "Yes" if prediction[0] == 1 else "No")
