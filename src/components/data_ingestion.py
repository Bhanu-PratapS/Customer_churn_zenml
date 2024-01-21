import os
import sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustmeException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transfromation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")
        try:
            logger.info("Data Reading using Pandas library from the local system")
            data = pd.read_csv(os.path.join("data", "census-income.csv"))
            logger.info("Data Reading completed")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Dat spliteted into train and test")
            
            train,test = train_test_split(data,test_size=0.3,random_state=42)
            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logger.info("Data Ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logger.info("Error occured in Data Ingestion")
            raise CustmeException(e,sys)
    if __name__ == "__main__":
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_training(train_arr, test_arr))
