# Lung Disease Prediction 🫁

This Python script uses a Random Forest Classifier to predict lung disease levels based on various health factors.
It is paired with a friendly web ui supported by streamlit

## Setup 
- Install pipenv to fetch all dependencies
```
pip install pipenv
```
```
git clone https://this_repository/cancer
cd cancer
pipenv install
```
- Clone the repository and install dependencies
- Activate the virutal environment
```
pipenv shell
```
- There are two files `magic_model.py` and `lung.py`, run streamlit run `lung.py` to start the web application
```
streamlit run lung.py
```


## Code Explanation

1. **Data Preprocessing**: The script first maps the 'Gender' column to numerical values (0 for 'Male' and 1 for 'Female'). Then, it uses pandas' `get_dummies` function to convert categorical variables into dummy/indicator variables. This is done for a list of specified columns in the DataFrame `df`.

2. **Feature Selection**: The script separates the target variable 'Level' from the features. This is done by dropping the 'Level' column from the DataFrame to create the features DataFrame `X`, and setting `y` as the 'Level' column.

3. **Train-Test Split**: The features and target variable are split into training and testing sets using the `train_test_split` function from Scikit-learn. The test size is set to 20% of the total data, and a random state is set for reproducibility.

4. **Model Training**: A Random Forest Classifier is created and fitted to the training data. The number of trees in the forest (`n_estimators`) is set to 100, and the random state is set for reproducibility.

5. **Feature Importance Extraction**: The feature importances are extracted from the trained model using the `feature_importances_` attribute. These represent how much each feature contributed to the model's predictions.

6. **Feature Importance Display**: The feature importances are stored in a DataFrame along with the corresponding feature names. The DataFrame is sorted by importance in descending order. This allows for easy visualization of which features were most important in the model's predictions.

## Libraries Used

- pandas: For data manipulation and analysis.
- sklearn.model_selection.train_test_split: For splitting the data into training and testing sets.
- sklearn.ensemble.RandomForestClassifier: For the Random Forest Classifier model.