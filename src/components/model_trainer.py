import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for GridSearchCV
            params = {
                "Decision Tree": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {"learning_rate": [0.1, 0.01, 0.05, 0.001], "n_estimators": [8, 16, 32, 64, 128, 256]},
                "CatBoosting Regressor": {"depth": [6, 8, 10], "learning_rate": [0.01, 0.05, 0.1], "iterations": [30, 50, 100]},
                "AdaBoost Regressor": {"learning_rate": [0.1, 0.01, 0.5, 0.001], "n_estimators": [8, 16, 32, 64, 128, 256]},
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params
            )

            # Get best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict on test set
            predicted = best_model.predict(X_test)
            from sklearn.metrics import r2_score
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R² score on test set: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
