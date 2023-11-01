import numpy as np
import cvxpy as cp
import pandas as pd


# ====================== Utility Functions ======================

def preprocessData(filename):
    '''
    Preprocesses the data by loading it, shuffling it, and dividing it into training and test sets.
    
    Parameters:
    filename (str): The path to the CSV file containing the data.
    
    Returns:
    tuple: A tuple containing the following elements:
        X_train (numpy array): The feature matrix for the training data.
        Y_train_lat (numpy array): The latitude values for the training data.
        Y_train_lon (numpy array): The longitude values for the training data.
        X_test (numpy array): The feature matrix for the test data.
        Y_test_lat (numpy array): The latitude values for the test data.
        Y_test_lon (numpy array): The longitude values for the test data.
    
    '''
     
    data = pd.read_csv(filename, delimiter='\t')

    # Using trainDataPct% of the data for training
    trainDataSize = int(0.7 * data.shape[0])
    trainData = data.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, :-1].values
    Y_train_lat = (trainData.iloc[:, -2].values).reshape(-1, 1)
    Y_train_lon = (trainData.iloc[:, -1].values).reshape(-1, 1)

    # Using testDataPct% of the data for testing 
    testData = data.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, :-1].values
    Y_test_lat = (testData.iloc[:, -2].values).reshape(-1, 1)
    Y_test_lon = (testData.iloc[:, -1].values).reshape(-1, 1)

    return X_train, Y_train_lat, Y_train_lon, X_test, Y_test_lat, Y_test_lon
