from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from typing import Protocol

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

class StructurePredictor(Protocol):
    def train(self):
        pass
    def load(self):
        pass
    def predict(self):
        pass
    def predict_and_compose(self):
        pass

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
        self.save_folder = Path(self.config['save_to']) / 'structure_predictor'
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder / "lookup_structures.pkl"

    
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
        
        # Save the models
        with open(self.save_as, 'wb') as f:
            pickle.dump((self.models, self.norm), f)

    def load(self, feature_table):
        self.feature_table = feature_table
        print("load lookup-structure models")
        with open(self.save_as, 'rb') as f:
            self.models, self.norm = pickle.load(f)
        return

    def _create_model_and_fit(self, X, y):
        model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**self.params))
        model.fit(X, y)
        return model

    def predict_and_compose(self, inputs: dict, X_base: np.ndarray, perturb_structure:bool=False):
        """ With the given inputs, give out the possible modes,
            and compose new temporal bases X_compose

        Parameters
        ----------
        inputs:  boundary conditions. The keys should be the same as self.config['x_labels']
        X_base: {array-like, with the shape (N_libarary_samples, N_libarary_modes)}
        perturb_structure: if True, do random sampling of structures based on its probability

        Returns
        -------
        X_compose: {array-like, with shape (self.config['N_modes'], self.config['N_t'])}
        predicted_features: {array-like, with shape (self.config['N_modes'], self.config['y_labels'])}, 
                            the features to be fed into the next machine

        """
        X_compose = np.zeros(X_base.shape[1:])

        if perturb_structure:
            predicted_df = self._unc_predict(inputs)
        else:
            predicted_df = self._predict(inputs)
        predicted_features = predicted_df[self.config['y_labels']].to_numpy()
        print(f" Predicted features shape = {predicted_features.shape}")

        # Compose predicted bases
        for i in predicted_df.index:
            sample = predicted_df.loc[i, 'sample']
            mode = predicted_df.loc[i, 'mode']
            X_compose[mode, :] = X_base[sample, mode, :]
        return X_compose, predicted_features

    def _predict(self, inputs: dict) -> pd.DataFrame:
        """ Given the inputs, deterministically predict the modes' feature table

        Parameters
        ----------
        inputs: {dict}, the keys should be the same as self.config['x_labels']

        Returns
        -------
        predicted_modes_features_as_table: {pd.DataFrame} of lenth: self.config['N_modes]
        """
        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        
        idx = []
        for mode in range(self.config['N_modes']):
            model = self.models[mode]
            y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]
            y_label_pred = model.predict(x)
            d = self._compare_distance(y_label_pred, y_label)
            id = y_label.index[d.argmin()]
            idx.append(id)
        predicted_modes_features_as_table = self.feature_table.iloc[idx]
        return predicted_modes_features_as_table

    def _unc_predict(self, inputs: dict):
        """ Given the inputs, PROBABILISTICALLY predict the modes' feature table

        Parameters
        ----------
        inputs: {dict}, the keys should be the same as self.config['x_labels']

        Returns
        -------
        predicted_modes_features_as_table: {pd.DataFrame} of lenth: self.config['N_modes]
        """ 
        modified_table = pd.DataFrame(self.feature_table)

        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        idx = []

        for mode in range(self.config['N_modes']):
            # y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]

            y_label = modified_table[modified_table['mode']== mode][self.config['y_labels']]

            # Get prediction
            model = self.models[mode]
            y_label_pred = model.predict(x)
            
            # modify table
            d = self._compare_distance(y_label_pred, y_label)
            prob = 1/d / sum(1/d)
            id_candidates = y_label.index

            # Sample those probability
            random_choice = np.random.choice(a=id_candidates, p=prob)
            idx.append(random_choice)
            modified_table.iloc[random_choice][self.config['y_labels']] = y_label_pred
        
        return modified_table.iloc[idx]
    def get_library_probability(self, inputs:dict, N_lib_samples: int, N_lib_modes: int, show=True):
        probs = np.zeros((N_lib_samples, N_lib_modes))
        for mode in range(self.config['N_modes']):
            probs[:, mode] = self._compute_prob_of_mode(inputs, mode)
        
        if show:
            fig, ax = plt.subplots(figsize=(8,4)) 
            sns.heatmap(probs, cmap='RdPu', vmin=0, vmax=1)
            plt.xlabel("Tempoeral Mode")
            plt.ylabel("Training ID")
            plt.show()
        return probs
    
    def _compare_distance(self, y_pred: np.ndarray, y_label: np.ndarray):
        """ The y_pred: shape (1, N_ylabels) """
        assert y_pred.shape == (1, len(self.config['y_labels']))
        return np.linalg.norm( y_label - y_pred, axis=1)
            
    def _compare_distance_and_return_id(self, y_pred: np.ndarray, y_label: np.ndarray):
        """ The y_pred: shape (1, N_ylabels) """
        distance = self._compare_distance(y_pred, y_label)
        id = distance.argmin() 
        return y_label.index[id]

    def _compute_prob_of_mode(self, inputs, mode):
        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]

        # Get prediction
        model = self.models[mode]
        y_label_pred = model.predict(x)
        
        # modify table
        d = self._compare_distance(y_label_pred, y_label)
        prob = 1/d / sum(1/d)
        return prob 
            
