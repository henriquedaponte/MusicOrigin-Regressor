import numpy as np
import cvxpy as cp
import pandas as pd


# ====================== Utility Functions ======================

def meanSquaredError(predicted, actual):
    '''
    Calculate the mean squared error (MSE) between predicted and actual values.
    
    Parameters:
    - predicted (numpy array): The array of predicted values generated by the model.
    - actual (numpy array): The array of actual or true values.
    
    Returns:
    - int: The mean squared error, rounded to the nearest whole number, between the predicted and actual values.
    
    Notes:
    The mean squared error is computed as the average of the squared differences between predicted and actual values.
    '''
    
   # Ensure that predicted array is not empty
    if len(predicted) == 0:
        raise ValueError("Predicted array is empty.")
        
    # Ensure that there are no NaN values in predicted and actual arrays
    if np.isnan(predicted).any() or np.isnan(actual).any():
        raise ValueError("NaN values detected in input arrays.")
    
    r = actual - predicted # Residuals
    mse = np.sum(r**2) / len(r)
    
    return mse

def preprocessData(filename, lat):
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
     
    data = pd.read_csv(filename, delimiter=',')

    # Using 70% of the data for training
    trainDataSize = int(0.7 * data.shape[0])
    trainData = data.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, :-2].values
    Y_train_lat = (trainData.iloc[:, -2].values).reshape(-1, 1)
    Y_train_lon = (trainData.iloc[:, -1].values).reshape(-1, 1)

    # Using 30% of the data for testing 
    testData = data.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, :-2].values
    Y_test_lat = (testData.iloc[:, -2].values).reshape(-1, 1)
    Y_test_lon = (testData.iloc[:, -1].values).reshape(-1, 1)

    if(lat):
        return X_train, Y_train_lat, X_test, Y_test_lat
    else:
        return X_train, Y_train_lon, X_test, Y_test_lon
    
def kFoldCrossValidation(filename, k, verbose=False):

    data = pd.read_csv(filename, delimiter=',')
    
    # Calculate the size of each fold
    fold_size = len(data) // k
    
    # Initialize a list to store the mean squared error for each fold
    mse_train_lat = 0
    mse_train_lon = 0
    mse_test_lat = 0
    mse_test_lon = 0
    i = 1
    
    # Perform k-fold cross-validation
    for fold in range(k):
    
        # Define the test data for this fold
        start = fold * fold_size
        end = start + fold_size if fold != (k - 1) else len(data)
        test_data = data.iloc[start:end]
        
        # Define the training data for this fold
        train_data = data.drop(test_data.index)

        # Aloocating proper data for training
        X_train = train_data.iloc[:, :-2].values
        Y_train_lat = (train_data.iloc[:, -2].values).reshape(-1, 1)
        Y_train_lon = (train_data.iloc[:, -1].values).reshape(-1, 1)

        # Allocating proper data for testing
        X_test = test_data.iloc[:, :-2].values
        Y_test_lat = (test_data.iloc[:, -2].values).reshape(-1, 1)
        Y_test_lon = (test_data.iloc[:, -1].values).reshape(-1, 1)
        
        if(verbose):
            print('=========================================================', '\n')
            print('                  MSE\'s for iteration', i,'k = ', k)
            print('=========================================================', '\n')
        # Training lattitude and longitude models here on training data
        a, b = testModelKFold(X_train, Y_train_lat, X_test, Y_test_lat, True, verbose)
        c, d = testModelKFold(X_train, Y_train_lon, X_test, Y_test_lon, False, verbose)

        mse_train_lat += a
        mse_test_lat += b
        mse_train_lon += c
        mse_test_lon += d
        i  += 1

    
    # Calculate the average MSE across all folds
    mse_train_lat = mse_train_lat / k
    mse_test_lat = mse_test_lat / k
    mse_train_lon = mse_train_lon / k
    mse_test_lon = mse_test_lon / k

    print("Final training set mean squared error for Latitude Model after k-fold cross validation (k = {}): {}".format(k, mse_train_lat))
    print("Final testing set mean squared error for Latitude Model after k-fold cross validation(k = {}): {}".format(k, mse_test_lat))
    print("\n")
    print("Final training set mean squared error for Longitutde Model after k-fold cross validation(k = {}): {}".format(k, mse_train_lon))
    print("Final testing set mean squared error for Longitutde Model after k-fold cross validation(k = {}): {}".format(k, mse_test_lon))
    print("\n")

def trainModel(X_train, Y_train):

    # Defiing decision variables
    beta = cp.Variable((X_train.shape[1], 1))

    # Defining objective function
    objective = cp.Minimize(cp.sum_squares(Y_train - X_train @ beta))

    # Formulating problem
    problem = cp.Problem(objective)

    # Solving problem
    problem.solve()

    return beta.value

def trainModelLassoLeastSquares(X_train, Y_train, lambdaCoeff):
    
        # Defiing decision variables
        beta = cp.Variable((X_train.shape[1], 1))
    
        # Defining objective function
        objective = cp.Minimize((cp.sum_squares(Y_train - X_train @ beta) + lambdaCoeff * cp.sum_squares(beta)) / X_train.shape[0])
    
        # Formulating problem
        problem = cp.Problem(objective)
    
        # Solving problem
        problem.solve()
    
        return beta.value

def trainModelLassoL1(X_train, Y_train, lambdaCoeff):
    
        # Defiing decision variables
        beta = cp.Variable((X_train.shape[1], 1))
    
        # Defining objective function
        objective = cp.Minimize((cp.sum_squares(Y_train - X_train @ beta) + lambdaCoeff * cp.norm(beta, 1)) / X_train.shape[0])
    
        # Formulating problem
        problem = cp.Problem(objective)
    
        # Solving problem
        problem.solve()
    
        return beta.value

def testModel(filename, lat):

    X_train, Y_train, X_test, Y_test = preprocessData(filename, lat)

    # Training model
    beta = trainModel(X_train, Y_train)

    # Predicting values for training set
    Y_pred_train = X_train @ beta

    # Calculating the mean squared error for training set
    mse_train = meanSquaredError(Y_pred_train, Y_train)

    # Predicting values for test set
    Y_pred_test = X_test @ beta

    # Calculating the mean squared error for test set
    mse_test = meanSquaredError(Y_pred_test, Y_test)

    if(lat):
        # Printing results
        print("Training set mean squared error for Latitude Model: {}".format(mse_train))
        print("Testing set mean squared error for Latitude Model: {}".format(mse_test))
        print("\n")
    else:
        # Printing results
        print("Training set mean squared error for Longitutde Model: {}".format(mse_train))
        print("Testing set mean squared error for Longitutde Model: {}".format(mse_test))
        print("\n")
    
    return mse_train, mse_test

def testModelLasso(filename, lat, leastSqu,lambdaCoeff, printResults = True, nonZero=False):

    # Question: Should I use the same formula for training and predicting?

    X_train, Y_train, X_test, Y_test = preprocessData(filename, lat)

    if(leastSqu):
        # Training model
        beta = trainModelLassoL1(X_train, Y_train, lambdaCoeff)
    else:
        beta = trainModelLassoLeastSquares(X_train, Y_train, lambdaCoeff)

    # Predicting values for training set
    Y_pred_train = (X_train @ beta)

    # Calculating the mean squared error for training set
    mse_train = meanSquaredError(Y_pred_train, Y_train)

    # Predicting values for test set
    Y_pred_test = X_test @ beta

    # Calculating the mean squared error for test set
    mse_test = meanSquaredError(Y_pred_test, Y_test)

    # Printing results
    if(printResults):
        print('Optimal Lambda: ', lambdaCoeff)
        if(lat):
            if(leastSqu):
                print("Training set mean squared error for Latitude Model using Lasso Regression regularized by Least Squares: {}".format(mse_train))
                print("Testing set mean squared error for Latitude Model using Lasso Regression regularized by Least Squares: {}".format(mse_test))
                print("\n")

            else:
                print("Training set mean squared error for Latitude Model using Lasso Regression regularized by L1: {}".format(mse_train))
                print("Testing set mean squared error for Latitude Model using Lasso Regression regularized by L1: {}".format(mse_test))
                print("\n")

        else:
            if(leastSqu):
                print("Training set mean squared error for Longitude Model using Lasso Regression regularized by Least Squares: {}".format(mse_train))
                print("Testing set mean squared error for Longitude Model using Lasso Regression regularized by Least Squares: {}".format(mse_test))
                print("\n")

            else:
                print("Training set mean squared error for Longitude Model using Lasso Regression regularized by L1: {}".format(mse_train))
                print("Testing set mean squared error for Longitude Model using Lasso Regression regularized by L1: {}".format(mse_test))
                print("\n")
    if(nonZero):
        nonZeroCoeff = 0
        for element in beta:
            if(element != 0):
                nonZeroCoeff += 1

        print('Number of non-zero coefficients: ', nonZeroCoeff)

    
    return mse_train, mse_test

def testModelKFold(X_train, Y_train, X_test, Y_test, lat, verbose = False):

    # Training model
    beta = trainModel(X_train, Y_train)

    # Predicting values for training set
    Y_pred_train = X_train @ beta

    # Calculating the mean squared error for training set
    mse_train = meanSquaredError(Y_pred_train, Y_train)

    # Predicting values for test set
    Y_pred_test = X_test @ beta

    # Calculating the mean squared error for test set
    mse_test = meanSquaredError(Y_pred_test, Y_test)

    if(verbose):
        if(lat):
            # Printing results
            print("Training set mean squared error for Latitude Model: {}".format(mse_train))
            print("Testing set mean squared error for Latitude Model: {}".format(mse_test))
            print("\n")
        else:
            # Printing results
            print("Training set mean squared error for Longitutde Model: {}".format(mse_train))
            print("Testing set mean squared error for Longitutde Model: {}".format(mse_test))
            print("\n")
    
    return mse_train, mse_test

def findOptimalLambda(filename, lat, lambda_range):
    best_lambda = None
    best_mse = float('inf')
    
    # Iterate over all the lambda values we want to test
    for lambdaCoeff in lambda_range:
        # Perform k-fold cross-validation
        # This is a simplified version, assuming you implement the k-fold inside this function
        avg_mse_train, avg_mse_test = testModelLasso(filename, 5, lambdaCoeff, lat, False)
        
        # Check if we got a better MSE with this lambda
        if avg_mse_test < best_mse:
            best_mse = avg_mse_test
            best_lambda = lambdaCoeff
            
    return best_lambda



# ====================== Problem Resolution ======================

# Problem 1.1
print('=========================================================', '\n')
print('                      Problem 1.1                        ', '\n')
print('=========================================================', '\n')
mse_train_lat, mse_test_lat = testModel('default_plus_chromatic_features_1059_tracks-1.txt', True)
mse_train_lon, mse_test_lon = testModel('default_plus_chromatic_features_1059_tracks-1.txt', False)

# Problem 1.2
print('=========================================================', '\n')
print('                      Problem 1.2                        ', '\n')
print('=========================================================', '\n')
kFoldCrossValidation('default_plus_chromatic_features_1059_tracks-1.txt', 10)
kFoldCrossValidation('default_plus_chromatic_features_1059_tracks-1.txt', 3)



# Problem 1.3
print('=========================================================', '\n')
print('                      Problem 1.3                        ', '\n')
print('=========================================================', '\n')
lambda_range = np.logspace(-4, 4, 100)  # for example, 100 values between 10^-4 and 10^4
best_lambda_lat = findOptimalLambda('default_plus_chromatic_features_1059_tracks-1.txt', True, lambda_range)
best_lambda_lon = findOptimalLambda('default_plus_chromatic_features_1059_tracks-1.txt', False, lambda_range)

testModelLasso('default_plus_chromatic_features_1059_tracks-1.txt', True, False, best_lambda_lat, nonZero=True)
testModelLasso('default_plus_chromatic_features_1059_tracks-1.txt', False, False, best_lambda_lon, nonZero=True)
testModelLasso('default_plus_chromatic_features_1059_tracks-1.txt', True, True, best_lambda_lat, nonZero=True)
testModelLasso('default_plus_chromatic_features_1059_tracks-1.txt', False, True, best_lambda_lon, nonZero=True)


