"""
Testing script that checks all the functions in churn_library script

Author: Aditya
Date: 22/01/2022
"""

from churn_library import import_data
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models

import logging
import sys
import glob
import os
import pytest
import joblib


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def df_loaded():
    """
    function to load initial data
    """
    try:
        df_loaded = import_data("data/bank_data.csv")
        logging.info("df_loaded - Initial dataset is loaded: SUCCESS")
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    return df_loaded


@pytest.fixture(scope="module")
def df_encoding(df_loaded):
    """
    function to encode the loaded dataframe
    """
    try:
        df_encoding = encoder_helper(df_loaded,
                                     category_lst=["Gender",
                                                   "Education_Level",
                                                   "Marital_Status",
                                                   "Income_Category",
                                                   "Card_Category"],
                                     response="Churn")
        logging.info("Encoding the dataframe: SUCCESS")
    except KeyError as err:
        logging.error(
            "Error in encoding: FAILURE")
        raise err

    return df_encoding


@pytest.fixture(scope="module")
def df_feature_engg(df_encoding):
    """
    function to perform feature engineering, does the train test split
    """
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_encoding, response="Churn")
        logging.info("Feature engineering: SUCCESS")
    except BaseException:
        logging.error(
            "Feature engineering: FAILURE")
        raise

    return x_train, x_test, y_train, y_test


def test_import(df_loaded):
    """
    testing if data is loaded correctly
    """
    try:
        assert df_loaded.shape[0] > 0
        assert df_loaded.shape[1] > 0
        logging.info("Import data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Data does not contain rows and columns")
        raise err


def test_eda(df_loaded):
    """
    testing if eda is done correctly
    """
    perform_eda(df_loaded)

    for image in ["Churn",
                  "Customer_Age",
                  "Marital_Status",
                  "Total_Trans_Ct",
                  "Heatmap"]:
        try:
            with open("images/eda/%s.jpg" % image, "r"):
                logging.info("EDA code testing: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Performing EDA : Failure")
            raise err


def test_feature_engg(df_feature_engg):
    """
    testing if feature engineering is done correctly, here we are checking if x and y are having same number of rows
    """
    try:
        assert len(df_feature_engg[0]) == len(df_feature_engg[2])
        logging.info("Feature engg: SUCCESS")
    except AssertionError as err:
        logging.error("Feature engg: FAILURE")
        raise err

    return df_feature_engg


def test_train_models(df_feature_engg):
    """
    testing the model training part
    """
    train_models(
        df_feature_engg[0],
        df_feature_engg[1],
        df_feature_engg[2],
        df_feature_engg[3])

    try:
        joblib.load("models/rfc_model.pkl")
        joblib.load("models/logistic_model.pkl")
        logging.info("Modelling: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Modelling: FAILURE")
        raise err

    for image in ["Logistic_Regression",
                  "Random_Forest",
                  "Feature_Importance",
                  "Roc_Curves"]:
        try:
            with open("images/results/%s.jpg" % image, 'r'):
                logging.info(
                    "Test results: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Test results: FAILURE")
            raise err


if __name__ == "__main__":
    pytest.main(["-s"])
