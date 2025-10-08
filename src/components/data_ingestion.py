import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging   # Ensure your logger.py is correctly set up
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            # Read dataset
            df = pd.read_csv(os.path.join("notebook", "data", "stud.csv"))
            logging.info("Read the dataset as dataframe")

            # Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Train-test split
            train_set, test_set = pd.DataFrame(), pd.DataFrame()
            train_set, test_set = df.sample(frac=0.8, random_state=42), df.drop(df.sample(frac=0.8, random_state=42).index)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")
            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error occurred in data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # ---------------- Data Ingestion ----------------
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print("✅ Train data path:", train_data_path)
        print("✅ Test data path:", test_data_path)

        # ---------------- Data Transformation ----------------
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print("✅ Preprocessor saved at:", preprocessor_path)

        # ---------------- Model Training ----------------
        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"\n✅ R² score of best model: {r2}")




    except Exception as e:
        print("❌ Something went wrong! Check logs for details.")
        import traceback
        traceback.print_exc()
