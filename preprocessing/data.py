import numpy as np
import joblib
from sklearn import preprocessing

class DataTransformer:
    @staticmethod
    def stationary_data(data, columns):
        satationary_df = data.copy()
        
        for column in columns:
            satationary_df[column] = np.log(satationary_df[column]) - np.log(satationary_df[column]).shift(1)

        satationary_df.dropna()
        
        return satationary_df
    
    @staticmethod
    def split_data(data):
        total_length = len(data)
        train_index = round(0.8 * total_length)
        
        train = data[:int(train_index), :]
        test = data[int(train_index):, :]
        
        return train, test
    
    @staticmethod
    def normalize_data(train, test):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        train = min_max_scaler.fit_transform(train)
        test = min_max_scaler.transform(test)
        joblib.dump(min_max_scaler, "./data/min_max_scaler.pkl")
        
        return train, test
    
    @staticmethod
    def formate_data(data, amount_of_features, seq_len):
        sequence_length = seq_len + 1 # index starting from 0
        result = []
        
        for index in range(len(data) - sequence_length): 
            result.append(data[index: index + sequence_length])
        
        result = np.array(result)
        
        x = result[:, :-1] 
        x = np.reshape(x, (x.shape[0], x.shape[1], amount_of_features))
        y = result[:, -1][:,-1] 

        return x, y