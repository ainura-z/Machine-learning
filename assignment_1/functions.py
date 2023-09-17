import pandas as pd
import numpy as np
import sklearn.metrics as sm

# function for scaling
def scaler(scaler_name, x_train, x_test):
    """ 
        This function scales our features by using scaler_name method.
        For example, scaler_name can be MinMaxScaler or StandartScaler, or any other.    
    """
    scaler = scaler_name()
    # fit it to the data.
    scaler.fit(x_train)
    # transform the data
    scaled_data_train = pd.DataFrame(scaler.transform(x_train), columns = x_train.columns)
    scaled_data_test = pd.DataFrame(scaler.transform(x_test), columns = x_test.columns)

    return scaled_data_train, scaled_data_test





# function for printing metrics for classifications
def performance_measurement_classification(y_test, y_pred):
    """ This function prints MSE, MAE, RMSE and R^2-Score metrics for train and test datasets"""

    print('Accuracy: ', sm.accuracy_score(y_test, y_pred))
    print('Precision: ', sm.precision_score(y_test, y_pred))
    print('Recall: ', sm.recall_score(y_test, y_pred))
    print("F1-Score:", sm.f1_score(y_test, y_pred))