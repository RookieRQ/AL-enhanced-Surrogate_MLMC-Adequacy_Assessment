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
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

class MachineLearning:
    """
    This class provides surrogate models, for storage case study.

    """

    def __init__(self, train_size, use_real_lol):
        """
        Initialise Machine Learning object
        Parameters:
            train_size : int
                number of training samples for HistGradientBoostingRegressor
            use_real_lol : Bool
                if True, use lol of training data to select samples training for SVR. 
                Otherwise, use lol prediction of HistGradientBoostingRegressor to select training samples for SVM.
        """
            
        self.lol_model, self.scaler, self.ens_model, self.lol_train_time, self.ens_train_time = self.fit(train_size, use_real_lol)
        
    def fit(self, train_size, use_real_lol):
        '''
        train AI models to predict LOL and ENS
        Parameters:
            train_size : int
                number of training samples for HistGradientBoostingRegressor
            use_real_lol : Bool
                if True, use lol of training data to select samples training for SVM. 
                Otherwise, use lol prediction of HistGradientBoostingRegressor to select training samples for SVR.
        Returns: 
            lol_model: object
                HistGradientBoostingRegressor model that predicts LOL
            scaler: object
                scaler object to normalize features for SVR
            ens_model: object
                SVR model that predicts ENS
            lol_train_time: float
                training time for LOL estimator
            ens_train_time: float
                training time for ENS estimator
        '''
        import time
        start_time = time.time()
        X_train = self.load_data(os.getcwd()+"/data/AIdata/daily_margin.csv")
        lol_train = self.load_data(os.getcwd()+"/data/AIdata/lol.csv")
        if train_size<=1:
            train_size = X_train.shape[0]*train_size
        lol_index = np.random.choice( X_train.shape[0],train_size, replace=False)
        X_train = X_train[lol_index,:]
        lol_train = lol_train[lol_index]
        if(X_train.shape[0]>2000):
            min_leaf = 20
        else:
            min_leaf = 5
        lol_model = HistGradientBoostingRegressor(random_state=0, min_samples_leaf=min_leaf).fit(X_train, lol_train)
        lol_train_time = time.time()-start_time
        start_time = time.time()
        if use_real_lol:
            ens_index = lol_train>0
        else:
            y_hat = np.rint(lol_model.predict(X_train))
            ens_index = y_hat>0
        X_train[X_train>1] = 1
        ens_train = self.load_data(os.getcwd()+"/data/AIdata/ens.csv")[lol_index]
        ens_train = ens_train[ens_index]
        X_train = X_train[ens_index,:]
        X_train, scaler = self.scale_data(X_train)
        ens_model = svm.SVR(cache_size=500, kernel='linear', epsilon=0.1, C=100, degree=2).fit(X_train, ens_train)
        ens_train_time = time.time()-start_time
        return lol_model, scaler, ens_model, lol_train_time, ens_train_time
    def scale_data(self, data):
        '''
        normalized data
        '''
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data), scaler

    
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
        if target==2 :            
            data[data>1] = 1
            data = self.scaler.transform(data)
            ens = self.ens_model.predict(data)
            ens[lol<1] = 0
        return lol, ens




if __name__ == "__main__":
    
    from sklearn.metrics import  mean_squared_error
    st = [500,1000,5000]
    for train_size in st:
        ML = MachineLearning(train_size= train_size, use_real_lol=True)
        X_test = ML.load_data(os.getcwd()+"/data/AIdata/daily_margin_test.csv")
        ens_test = ML.load_data(os.getcwd()+"/data/AIdata/ens_test.csv")
        lol_test = ML.load_data(os.getcwd()+"/data/AIdata/lol_test.csv")
        lol_hat, ens_hat = ML.predict(X_test)
        print(f"root_mean_squared_error(LOL) : {mean_squared_error( lol_test, lol_hat, squared=False):.4f}")
        print(f"root_mean_squared_error(ENS) : {mean_squared_error( ens_test, ens_hat, squared=False):.4f}")
