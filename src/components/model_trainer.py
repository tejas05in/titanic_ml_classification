import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import os , sys

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            model = SVC()
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test,y_pred)
            logging.info(f"accuracy score of SVC: {score*100:.2f}%")
            print(f"accuracy score of SVC: {score*100:.2f}%")
            

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )
            
            logging.info("Model training complete and model saved in artifacts as model.pkl")
        except Exception as e:
            raise CustomException(e,sys)