# src/utils.py

from src.logger import logging
from src.exception import CustomException
import os,sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            parameters = params[model_name]
            
            # Using GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(model, parameters, cv=5, scoring="accuracy")
            grid_search.fit(X_train, y_train)
            
            # Setting the best hyperparameters to the model
            model.set_params(**grid_search.best_params_)
            
            # Fitting the model on the training data
            model.fit(X_train, y_train)
            
            # Making predictions on the test data
            y_pred = model.predict(X_test)
            
            # Calculating accuracy score
            test_model_accuracy = accuracy_score(y_test, y_pred)
            
            # Storing the accuracy score in the report dictionary
            report[model_name] = test_model_accuracy
        
        return report
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)
