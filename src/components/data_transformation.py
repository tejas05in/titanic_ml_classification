from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OneHotEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("preprocessor creation started")
            
            # Categorical and numerical columns
            categorical_cols = ['pclass','sex','cabin','embarked','title']
            numerical_cols = ['age', 'sibsp', 'parch', 'fare', 'family_size']
            

            logging.info("Data transformation pipeline initiated")

            ## Numerical Pipeline
            num_pipeline=Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
            steps=[
                ('ordinalencoder',OneHotEncoder(drop='first',handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ]

            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info("Preprocessor created and returned")


            return preprocessor


        except Exception as e:
            logging.info("Exception occured in Data Transformation")
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            logging.info("Data Transformation Initiated")
            train_input_feature_df = pd.read_csv(train_data_path)
            test_input_feature_df = pd.read_csv(test_data_path)
            input_feature_df = pd.concat([train_input_feature_df.assign(ind='train'), test_input_feature_df.assign(ind='test')])
            
            logging.info("Read train and test data completed")
            logging.info(f'Train Datatrame Head : \n {train_input_feature_df.head().to_string()}')
            logging.info(f'Test Datatrame Head : \n {test_input_feature_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column = 'survived'
            drop_columns = [target_column,'boat', 'body', 'home.dest','ticket']
            
            


            ##dividing the dataset into independent and dependent features
            ## Training data
            target_feature_train_df = input_feature_df[input_feature_df['ind'].eq('train')][target_column]
            target_feature_test_df = input_feature_df[input_feature_df['ind'].eq('test')][target_column]
            input_feature_df = input_feature_df.drop(columns=drop_columns,axis=1)
            
            
            ## missing values
            input_feature_df['fare'] = input_feature_df['fare'].fillna(input_feature_df.fare.median())
            
            ## feature engineering for age null values imputation
            input_feature_df['title'] = input_feature_df['name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()
            normalized_titles = {
                "Capt":       "Officer",
                "Col":        "Officer",
                "Major":      "Officer",
                "Jonkheer":   "Royalty",
                "Don":        "Royalty",
                "Sir" :       "Royalty",
                "Dr":         "Officer",
                "Rev":        "Officer",
                "the Countess":"Royalty",
                "Dona":       "Royalty",
                "Mme":        "Mrs",
                "Mlle":       "Miss",
                "Ms":         "Mrs",
                "Mr" :        "Mr",
                "Mrs" :       "Mrs",
                "Miss" :      "Miss",
                "Master" :    "Master",
                "Lady" :      "Royalty"
            }
            input_feature_df['title'] = input_feature_df['title'].map(normalized_titles)
            # dropping name column
            input_feature_df.drop(['name'],axis=1,inplace=True)
            # impute missing Age values using median of Title groups
            title_ages = dict(input_feature_df.groupby(['title'])['age'].median())
            # create a column of the average ages
            input_feature_df['age_med'] = input_feature_df['title'].apply(lambda x : title_ages[x])
            # filling the null age with age_med
            input_feature_df['age'].fillna(input_feature_df['age_med'], inplace=True)
            # age_med removal
            input_feature_df.drop(['age_med'],axis=1,inplace=True)
            
            # Embarked missing value imputation
            input_feature_df['embarked'].bfill(inplace=True)
            
            #Keeping cabin missing values as separate category = U
            input_feature_df['cabin'] = input_feature_df['cabin'].fillna('U')
            
            ## Feature Engineering
            # size of families (including the passenger)
            input_feature_df['family_size'] = input_feature_df['sibsp'] + input_feature_df['parch'] + 1
            # map first letter of cabin to itself
            input_feature_df['cabin'] = input_feature_df['cabin'].map(lambda x: x[0])
            
            # Reduce the dtypes to its optimal variant
            input_feature_df = input_feature_df.convert_dtypes()
            
            # separate input_feature_df into train and test
            input_feature_train_df = input_feature_df[input_feature_df['ind'].eq('train')]
            input_feature_test_df = input_feature_df[input_feature_df['ind'].eq('test')]
            
            # drop the ind column
            input_feature_train_df=input_feature_train_df.drop(['ind'],axis=1)
            input_feature_test_df=input_feature_test_df.drop(['ind'],axis=1)
            # input_feature_test_df.to_csv('artifacts\input_test_data.csv',index=False)
           
            # Data transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing dataset")
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj

            )
            logging.info("Data Transformation Completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:

            raise CustomException(e,sys)
        
