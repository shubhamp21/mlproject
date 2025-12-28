import os
import sys
import pandas as pd
import numpy as np
import dill

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object) -> None:
    '''
    This function saves a Python object to the specified file path using joblib.
    '''
    try:
        import joblib
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    This function evaluates multiple machine learning models and returns their R2 scores.
    '''
    from sklearn.metrics import r2_score

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            # Predict on training data
            y_train_pred = model.predict(X_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            train_model_score = r2_score(y_train, y_train_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> object:
    '''
    This function loads a Python object from the specified file path using joblib.
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys)
    
