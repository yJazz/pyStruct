from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from collections.abc import Iterable

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

import sys
sys.path.insert(0, r"F:\project2_phD_bwrx\code\drivers")

class LookupStructure:
    def __init__(self, df, x_labels, y_labels):
        self.N_samples = len(df['sample'].unique())
        self.N_modes = len(df['mode'].unique())
        self.normalizer_x = StandardScaler()
        self.normalizer_y = StandardScaler()
        self.df = df
        self.x_labels = x_labels
        self.y_labels = y_labels

        self.X = self.normalizer_x.fit_transform(df[x_labels])
        self.y = self.normalizer_y.fit_transform(df[y_labels])

    def create_model_and_fit(self, params):
        self.model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**params))
        self.model.fit(self.X, self.y)
        return

    def predict(self, input_df):
        X = self.normalizer_x.transform(input_df)
        return self.model.predict(X)

    def compute_mode_distances(self, mode, input_df):
        y_pred = self.predict(input_df)
        df = self.df.copy()
        df = df[df['mode'] == mode]
        y_ref = self.normalizer_y.transform(df[self.y_labels])
        # Get distances
        df['distance'] = [mean_squared_error(y_pred[0], r) for r in y_ref]
        return df

    def get_mode_deterministic_from_library(self, mode, input_dict):
        input_df = pd.DataFrame(input_dict, index=[0])
        df = self.compute_mode_distances(mode, input_df)
        return df.iloc[df['distance'].argmin(), :]





def split_train_test(wp, train_id):
    bool_series = wp['sample'].isin(train_id)
    wp_train = wp[bool_series]
    wp_test = wp[bool_series==False]
    return wp_train, wp_test


def get_structure_predictors_for_each_mode(wp_table_path, x_labels, y_labels, train_id=None):
    assert Path(wp_table_path).exists()
    wp =  pd.read_csv(wp_table_path)

    train_id = range(15)
    if train_id:
        wp_train, wp_test = split_train_test(wp, train_id)
    else:
        wp_train = wp
    # x_labels=['m_c', 'vel_ratio', 'theta_deg']
    # y_labels=['spatial_strength', 'temporal_behavior', 'singular']

    sps = []
    for mode in range(20):
        wp_mode = wp_train[wp_train['mode'] == mode]
        params = {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 1 }
        sp = LookupStructure(wp_mode, x_labels, y_labels)
        sp.create_model_and_fit(params)
        sps.append(sp)

    return sps

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
if __name__ == "__main__":
    # Get the weights property table
    wp_table_path = r"F:\project2_phD_bwrx\code\drivers\linear_weights_prediction\data\wp_0_15.csv"