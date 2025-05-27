"""
Surrogate model
@author: Ensieh Sharifnia
Delft University of Technology
e.sharifnia@tudelft.nl

Function optimal_n_store_generator() by Michael Evans, Imperial College London.

This code implements training surrogate model in 
"Multilevel Monte Carlo with Surrogate Models forResource Adequacy Assessment",
Ensieh Sharifnia and Simon Tindemans,
accepted for publication at PMAPS 2022.
A preprint is available at: arXiv:2203.03437

If you use (parts of) this code, please cite the preprint or published paper.
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MachineLearning:
    """
    This class provides surrogate models, for storage case study.

    """

    def __init__(self, train_size):
        """
        Initialise Machine Learning object
        Parameters:
            train_size : int
                number of training samples
        """
            
        self.lol_model, self.ens_model, self.lol_train_time, self.ens_train_time = self.fit(train_size)
        

    def fit(self, train_size):
        '''
        train AI models to predict LOL and ENS
        Parameters:
            train_size : int
                number of training samples
        Returns: 
            lol_model: object
                Random forest model that predicts LOL
            scaler: object
                scaler object to normalize features for SVR
            ens_model: object
                Random forest model that predicts ENS
            lol_train_time: float
                training time for LOL estimator
            ens_train_time: float
                training time for ENS estimator
        '''
        import time
        start_time = time.time()
        X_train = self.load_data("../data/AIdata/daily_margin_test.csv")
        lol_train = self.load_data("../data/AIdata/lol_test_daily.csv")
        ens_train = self.load_data("../data/AIdata/ens_test_daily.csv")

        if train_size<=1:
            train_size = X_train.shape[0]*train_size

        lol_index = np.random.choice(X_train.shape[0], train_size, replace=False)
        X_train = X_train[lol_index,:]
        lol_train = lol_train[lol_index]

        lol_model = RandomForestRegressor().fit(X_train, lol_train)
        lol_train_time = time.time()-start_time

        start_time = time.time()
        ens_train = ens_train[lol_index]

        ens_model = RandomForestRegressor().fit(X_train, ens_train)
        ens_train_time = time.time()-start_time

        return lol_model, ens_model, lol_train_time, ens_train_time
    
    
    def load_data(self, file_name):
        '''
        load data from the file_name
        Parameters: 
            file_name: string
                file address
        Returns:
            data: ndarray
                an array of file's content
        '''
        data = np.ascontiguousarray(np.genfromtxt(file_name, delimiter=','))
        return data


    def predict(self, data, target = 2):
        '''
        predict LOL /and ENS for given data
        Parameters:
            data: ndarray
                each row is a data to predict
            target: int
                if target = 2, then provide lol and ens. otherwise just lol
        Returns:
            lol: ndarray
                LOL prediction
            ens: ndarray
                ENS prediction
        '''
        lol = np.rint(self.lol_model.predict(data))
        ens = None

        if target==2:
            ens = self.ens_model.predict(data)

        return lol, ens




if __name__ == "__main__":
    
    from sklearn.metrics import root_mean_squared_error
    st = [500,1000,5000]
    for train_size in st:
        ML = MachineLearning(train_size=train_size)
        X_test = ML.load_data("../data/AIdata/daily_margin_test.csv")
        ens_test = ML.load_data("../data/AIdata/ens_test_daily.csv")
        lol_test = ML.load_data("../data/AIdata/lol_test_daily.csv")
        lol_hat, ens_hat = ML.predict(X_test)
        print(f"root_mean_squared_error(LOL) : {root_mean_squared_error(lol_test, lol_hat):.4f}")
        print(f"root_mean_squared_error(ENS) : {root_mean_squared_error(ens_test, ens_hat):.4f}")
