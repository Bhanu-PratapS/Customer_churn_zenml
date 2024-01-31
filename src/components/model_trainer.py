import os, sys
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_object, evaluate_model
from sklearn.model_selection import train_test_split

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")

@dataclass
class ModelTrainerConfig:
    train_model_folder = "artifacts/model_trainer"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting our data into dependent and independent variables")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic": LogisticRegression()
            }
            params = {
                "Random Forest": {
                    "class_weight": ["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ['gini', 'entropy', 'log_loss'],
                    "splitter": ['best', 'random'],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4, 5],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["auto", "sqrt", "log2"]
                },
                "Logistic": {
                    "class_weight": ["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }

            # Use train_test_split to create a validation set
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            for model_name, model in models.items():
                grid_search = GridSearchCV(model, params[model_name], scoring='accuracy', cv=5)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Evaluate the best model
                model_report = evaluate_model({model_name: best_model}, {}, X_train, y_train, X_test, y_test)
                best_model_score = model_report[model_name]

                print(f"Best {model_name} Model Found, Accuracy Score: {best_model_score}")
                logging.info(f"Best {model_name} model found, Accuracy Score: {best_model_score}")

                # Save the best model using pickle
                save_object(file_path=os.path.join(self.model_trainer_config.train_model_folder, f"{model_name}_model.pkl"), obj=best_model)
                logging.info(f"{model_name} Model Saved using pickle")

        except Exception as e:
            raise CustomException(e, sys)


from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting our data into dependent and independent variables")
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            model = {
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Logistic":LogisticRegression()
            }
            params = {
                "Random Forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }
            model_report:dict = evaluate_model(model,params,X_train,y_train,X_test,y_test,models=model,params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name  = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = model[best_model_name]
            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            logging.info(f"best model found,Model Nmae is: {best_model_name},Accuracy_Score: {best_model_score}")
            save_object(file_path=self.model_trainer_config.train_model_file_path,obj=best_model)
            logging.info("Model Saved")
        except Exception as e:
            raise CustomException(e,sys)
            