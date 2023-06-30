import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
        
    def initiate_data_ingestion(self):
        try:
            
            logging.info("Initiating data ingestion")
            data = pd.read_csv(os.path.join('notebooks/data', 'income_cleandata.csv')) #reading this csv file with the help of notebooks/data folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # create artifacts directory/folder to save the raw data, train data and test data
            data.to_csv(self.ingestion_config.raw_data_path, index=False) #saving raw_data.csv file in artifact directory
            
            train_set, test_set = train_test_split(data, test_size=0.3, random_state=42) #splitting data into train test
            logging.info("data splitted into train and test")
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True) #saving train dataset
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True) #saving test dataset
            
            logging.info("Data ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occur in data ingestion stage")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()        