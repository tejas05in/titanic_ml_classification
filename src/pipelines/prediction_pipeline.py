import sys , os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

## Input data features and their data types
# pclass                  Int64
# sex            string[python]
# age                   Float64
# sibsp                   Int64
# parch                   Int64
# fare                  Float64
# cabin          string[python]
# embarked       string[python]
# title          string[python]
# family_size             Int64

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scled = preprocessor.transform(features)

            pred = model.predict(data_scled)
            return pred        
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 pclass:int,
                 sex:str,
                 age:float,
                 sibsp:int,
                 parch:int,
                 fare:float,
                 cabin:str,
                 embarked:str,
                 title:str,
                 ):
        
        self.pclass = pclass
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch
        self.fare = fare
        self.cabin = cabin
        self.embarked = embarked
        self.title = title
        

    def get_data_as_dataframe(self):
        try:
            self.family_size = self.sibsp + self.parch + 1
            custom_data_input_dict = {
                'pclass':[self.pclass],
                "sex" :[self.sex],
                "age" : [self.age],
                "sibsp" : [self.sibsp],
                "parch":[self.parch],
                "fare":[self.fare],
                "cabin" : [self.cabin],
                "embarked" : [self.embarked],
                "title":[self.title],
                "family_size": [self.family_size],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            raise CustomException(e,sys)