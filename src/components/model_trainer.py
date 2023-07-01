import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components import data_transformation
from src.components.data_transformation import DataTransformation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report,f1_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils import evaluate_model
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            
            logging.info("Splitting data into dependent and independent features")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, :-1],
                test_arr[:, -1],
                test_arr[:, -1]
            )
            
            logging.info("training data on various models and parameters")
            model = {
                "random Forest" : RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Logistic Regression" : LogisticRegression()
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
                "Logistic Regression":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, 
                                               models = model, params = params)
             
            #to get best model from our report dict    
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = model[best_model_name]
            
            print(f"best model found,Model name is {best_model_name}, accuracy score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"best model found, Model Name is {best_model_name}, accuracy Score: {best_model_score}")

            
            save_obj(file_path=self.model_trainer_config.train_model_file_path,
                     obj=best_model)
            
        except Exception as e:
            raise CustomException(e, sys)
        