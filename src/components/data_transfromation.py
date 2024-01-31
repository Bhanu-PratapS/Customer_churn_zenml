import os, sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass 
class DataTransformationConfigs:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocess.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfigs()

    def get_data_transformation_obj(self):
        try:
            logger.info("Data Transformation started")
            numerical_features = ['age', 'education.num', 'hours.per.week']
            cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, df):
        try:
            numerical_features = ['age', 'education.num', 'hours.per.week']
            for col in numerical_features:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                IQR = q3 - q1
                lower_limit = q1 - 1.5 * IQR
                upper_limit = q3 + 1.5 * IQR
                df = df[(df[col] > lower_limit) & (df[col] < upper_limit)]
            return df
        except Exception as e:
            logger.info("Outliers Handling code")
            raise CustomException(e, sys)

    def inititate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            
            numerical_features = ['age', 'education.num', 'hours.per.week']
            categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income']
            
            train_data = self.remove_outliers_IQR(train_data)
            test_data = self.remove_outliers_IQR(test_data)
            preprocessor = self.get_data_transformation_obj()
            
            target_column = ["income"]
            drop_columns = [target_column]
            
            logger.info("Splitting train data into dependent and independent features")
            
            input_features_train_data = train_data.drop(drop_columns, axis=1)
            target_features_train_data = train_data[target_column]
            
            logger.info("Splitting train data into dependent and independent features")
            
            input_feature_test_data = test_data.drop(drop_columns, axis=1)
            target_features_test_data = test_data[target_column]
            
            input_train_arr = preprocessor.fit_transform(input_features_train_data)
            input_test_arr = preprocessor.transform(input_feature_test_data)
            
            train_array = np.concatenate((input_train_arr, target_features_train_data), axis=1)
            test_array = np.concatenate((input_test_arr, target_features_test_data), axis=1)
            
            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocess_obj_file_path), exist_ok=True)
            
            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path, obj=preprocessor)
            return train_array, test_array, self.data_transformation_config.preprocess_obj_file_path
        except Exception as e:
            logger.info("Error occurred in Data Transformation")
            raise CustomException(e, sys)

