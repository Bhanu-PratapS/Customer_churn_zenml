import os, sys
from src.logger import logging
from src.exception import CustmeException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(slef):
    
        def predict(self,features):
            preprocessro_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
            model_path = os.path.join("artifacts/model_trainer","model.pkl")
            processor = load_object(preprocessro_path)
            model = load_object(model_path)
            scaled = processor.transform(features)
            pred = model.predict(scaled)
            return pred
    
class CustomClass:
    def __init__(self,
                age: int,
                education_num: int,
                hours_per_week: int,
                workclass: str,
                education: str,
                marital_status: str,
                occupation: str,
                relationship: str,
                race: str,
                sex: str,
                income: str):
        self.age = age
        self.education_num = education_num
        self.hours_per_week = hours_per_week
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.income = income
    def get_data_DataFrame(self):
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
                "race":[self.race],
                "sex":[self.sex],
                "income":[self.income]
                
            }
            data = pd.DataFrame(custom_input)
            return data
        except Exception as e:
            raise CustmeException(e,sys)