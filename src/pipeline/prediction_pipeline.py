import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:  
    def predict(self, features):
        preprocessor_filename = "preprocessor.pkl"
        model_filename = "model.pkl"
        
        preprocessor_path = os.path.join("artifacts", "data_transformation", preprocessor_filename)
        model_path = os.path.join("artifacts", "model_trainer", model_filename)
        
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        
        scaled = preprocessor.transform(features)
        pred = model.predict(scaled)
        return pred
@dataclass
class CustomClass:
    age: int
    education_num: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    income: str

    def get_data_dataframe(self):  # Fix the method name
        try:
            custom_input = {
                "age": [self.age],
                "education_num": [self.education_num],
                "hours_per_week": [self.hours_per_week],
                "workclass": [self.workclass],
                "education": [self.education],
                "marital_status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "income": [self.income]
            }
            data = pd.DataFrame(custom_input)
            return data
        except Exception as e:
            raise CustomException(e, sys)