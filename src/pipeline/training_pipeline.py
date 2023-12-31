import sys, os
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    
    #data ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    
    #data transfgormation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    #model trainer
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)