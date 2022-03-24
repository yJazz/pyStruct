from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

import sys
from pyStruct.data.dataset import read_pickle



def visualize_libray_matrix(sps, m_c, vel_ratio, theta_deg):
    library_matrix = np.zeros((20, 20))
    for mode in range(20):
        m = sps[mode].get_mode_deterministic_from_library(mode, m_c, vel_ratio, theta_deg)
        sample = int(m['sample'])
        library_matrix[sample, mode] = 1
        
        
    fig, ax = plt.subplots(figsize=(8,4)) 
    sns.heatmap(library_matrix, cmap='RdPu', vmin=0, vmax=1)
    plt.xlabel("Tempoeral Mode")
    plt.ylabel("Training ID")
    plt.title("Inlet: $m_h$=%.2lf, $m_c$=%.2lf  kg/s"%(m_c/vel_ratio, m_c))
    plt.show(block=False)
    return 


class LookupStructure:
    """ Given x_labels, output y_labels
        stratified by modes: each mode: a regression model
    """

    def __init__(self, config):
        self.config = config
        self.params = {
            'n_estimators': 100, 
            'max_depth': 7, 
            'min_samples_split': 2, 
            'min_samples_leaf': 1 
            }

        self.models = []
        self.norm = StandardScaler()
    
    def train(self, feature_table):
        self.feature_table = feature_table
        features = feature_table[self.config['x_labels']]
        features_x = self.norm.fit_transform(features)

        targets = feature_table[self.config['y_labels']].to_numpy()


        for mode in range(self.config['N_modes']):
            print(f'train structure predictor mode {mode}')
            x = features_x[feature_table['mode'] == mode, :]
            y = targets[feature_table['mode'] == mode, :]

            model = self._create_model_and_fit(x, y)
            self.models.append(model)

    def _create_model_and_fit(self, X, y):
        model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**self.params))
        model.fit(X, y)
        return model

    def predict(self, inputs: dict):
        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        
        idx = []
        for mode in range(self.config['N_modes']):
            model = self.models[mode]
            y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]
            y_label_pred = model.predict(x)
            id = self._compare_distance_and_return_id(y_label_pred, y_label)
            idx.append(id)
        return self.feature_table.iloc[idx]
            
    def _compare_distance_and_return_id(self, y_pred: np.ndarray, y_label: np.ndarray):
        """ The y_pred: shape (1, N_ylabels) """
        assert y_pred.shape == (1, len(self.config['y_labels']))

        distance = np.linalg.norm( y_label - y_pred, axis=1)
        id = distance.argmin() 
        return y_label.index[id]

    def predict_and_compose(self, inputs: dict, X_base: np.ndarray):
        """ With the given inputs, give out the possible modes,
            and compose new temporal bases X_compose
        """
        predicted_df = self.predict(inputs)

        # Compose predicted bases
        X_compose = np.zeros(X_base.shape[1:])

        for i in predicted_df.index:
            sample = predicted_df.loc[i, 'sample']
            mode = predicted_df.loc[i, 'mode']
            X_compose[mode, :] = X_base[sample, mode, :]
        return X_compose, predicted_df[self.config['y_labels']].to_numpy()
