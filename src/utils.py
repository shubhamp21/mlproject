import os
import sys
import pandas as pd
import numpy as np
import dill

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