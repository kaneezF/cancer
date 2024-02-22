import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT_PATH = os.path.dirname(__file__)


class MagicModel:
    """
    A class representing a magic model for predicting lung cancer.

    Attributes:
        df (pandas.DataFrame): The loaded dataset.
        X_train (pandas.DataFrame): The training features.
        X_test (pandas.DataFrame): The testing features.
        y_train (pandas.Series): The training labels.
        y_test (pandas.Series): The testing labels.
        rf_model (RandomForestClassifier): The random forest model for feature selection.
        rf_model_selected (RandomForestClassifier): The random forest model with selected features.

    Methods:
        load_data(): Loads the dataset and preprocesses it.
        split_data(): Splits the data into training and testing sets.
        train_model(): Trains the random forest model on the training data.
        get_feature_importances(): Calculates the feature importances.
        select_top_k_features(k): Selects the top k features based on importance.
        select_features_based_on_threshold(threshold): Selects features based on a threshold.
        select_features(features): Selects the specified features.
        train_model_with_selected_features(X_train_selected): Trains the model with selected features.
        evaluate_model_performance(X_test_selected): Evaluates the model performance on the testing data.
        predict_lung_cancer(input_data): Predicts lung cancer based on input data.
    """

    def __init__(self):
        self.df = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self):
        """
        Loads the dataset and preprocesses it.

        Returns:
            pandas.DataFrame: The preprocessed dataset.
        """
        dataset = os.path.join(ROOT_PATH, "data", "lungcancer.csv")
        df = pd.read_csv(dataset)
        df.drop(["index", "Patient Id", "Clubbing of Finger Nails"], axis=1, inplace=True)
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
        return df

    def split_data(self):
        """
        Splits the data into training and testing sets.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test.
        """
        X = self.df.drop("Level", axis=1)
        y = self.df["Level"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self):
        """
        Trains the random forest model on the training data.
        """
        self.rf_model.fit(self.X_train, self.y_train)

    def get_feature_importances(self):
        """
        Calculates the feature importances.

        Returns:
            pandas.DataFrame: A DataFrame containing the feature importances.
        """
        feature_importances = self.rf_model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": self.X_train.columns, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
        return feature_importance_df

    def select_top_k_features(self, k):
        """
        Selects the top k features based on importance.

        Args:
            k (int): The number of top features to select.

        Returns:
            list: A list of the top k features.
        """
        feature_importance_df = self.get_feature_importances()
        top_k_features = feature_importance_df.head(k)["Feature"].tolist()
        return top_k_features

    def select_features_based_on_threshold(self, threshold):
        """
        Selects features based on a threshold.

        Args:
            threshold (float): The importance threshold.

        Returns:
            list: A list of selected features.
        """
        feature_importance_df = self.get_feature_importances()
        selected_features = feature_importance_df[
            feature_importance_df["Importance"] >= threshold
        ]["Feature"].tolist()
        return selected_features

    def select_features(self, features):
        """
        Selects the specified features.

        Args:
            features (list): A list of features to select.

        Returns:
            tuple: A tuple containing the selected training and testing features.
        """
        X_train_selected = pd.DataFrame(self.X_train, columns=features)
        X_test_selected = pd.DataFrame(self.X_test, columns=features)
        return X_train_selected, X_test_selected

    def train_model_with_selected_features(self, X_train_selected):
        """
        Trains the model with selected features.

        Args:
            X_train_selected (pandas.DataFrame): The selected training features.
        """
        self.rf_model_selected.fit(X_train_selected, self.y_train)

    def evaluate_model_performance(self, X_test_selected):
        """
        Evaluates the model performance on the testing data.

        Args:
            X_test_selected (pandas.DataFrame): The selected testing features.

        Returns:
            float: The accuracy of the model.
        """
        y_pred_selected = self.rf_model_selected.predict(X_test_selected)
        accuracy_selected = accuracy_score(self.y_test, y_pred_selected)
        return accuracy_selected

    def predict_lung_cancer(self, input_data):
        """
        Predicts lung cancer based on input data.

        Args:
            input_data (array-like): The input data for prediction.

        Returns:
            array-like: The predicted lung cancer values.
        """
        prediction = self.rf_model.predict(input_data)
        return prediction


if __name__ == "__main__":
    model = MagicModel()
    model.train_model()
    top_k_features = model.select_top_k_features(10)
    X_train_selected, X_test_selected = model.select_features(top_k_features)
    model.train_model_with_selected_features(X_train_selected)
    accuracy_selected = model.evaluate_model_performance(X_test_selected)

    print("Accuracy with selected features:", accuracy_selected)
    user_input = [50, 2, 2, 2, 2, 2, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 2, 1, 2]
    other_features = [0] * (model.X_train.shape[1] - len(user_input))

    model_input = np.array([user_input + other_features])
    prediction = model.predict_lung_cancer(model_input)
    print("Prediction:", prediction)
