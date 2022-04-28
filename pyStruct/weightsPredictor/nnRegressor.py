from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import StandardScaler

from pyStruct.weightsPredictor.regressors import WeightsPredictor

class NnRegressor(WeightsPredictor):
    """Give feature"""
    def __init__(self, config: dict):

        self.config = config

        self.save_folder = Path(self.config['save_to']) / 'NN_weights_predictor'
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder / "NN_regressor"

    def _create_model(self):
        dropout_rate = 0.25
        inputs = Input(shape=(len(self.config['y_labels']),))
        x = Dense(64, activation='relu')(inputs)
        # x = Dropout(dropout_rate)(x, training=True)
        x = Dense(128, activation='relu')(x)
        # x = Dropout(dropout_rate)(x, training=True)
        x = Dense(256, activation='relu')(x)
        # x = Dropout(dropout_rate)(x, training=True)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs, outputs)    

        return model

    def _pre_processing(self, feature_table: pd.DataFrame):
        dataset = feature_table.copy()
        # Normalize 
        self.normalizer = StandardScaler()
        norm_features = self.normalizer.fit_transform(dataset[self.config['y_labels']])
        for i, label in enumerate(self.config['y_labels']):
            dataset[f'{label}'] = norm_features[:, i]

        # Split into train, test 
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # Specify features and labels
        train_features = train_dataset[self.config['y_labels']]
        test_feature = test_dataset[self.config['y_labels']]

        train_labels = train_dataset['w']
        test_labels = test_dataset['w']
        return train_features, train_labels, test_feature, test_labels
    
    def train(self, feature_table: pd.DataFrame, create_model=True, epochs=1000):
        train_features, train_labels, test_feature, test_labels = self._pre_processing(feature_table)
        if create_model:
            self.model = self._create_model()
            opt = keras.optimizers.Adam(learning_rate=0.005)
            self.model.compile(loss='mean_squared_error', optimizer=opt)
        self.model.fit(train_features, train_labels, epochs=epochs, validation_split=0.2)

        # Save 
        self.model.save(self.save_as)
        with open(self.save_as/'norm.pkl', 'wb') as f:
            pickle.dump(self.normalizer, f)
        return

    def load(self, feature_table):
        print("load file")
        self.model = keras.models.load_model(self.save_as)
        with open(self.save_as/'norm.pkl', 'rb') as f:
            self.normalizer = pickle.load(f)

    def predict(self, features: np.ndarray, samples=100):
        # Normalize features
        features = self.normalizer.transform(features)
        # predictions = []
        # for i in range(samples):
            # predictions.append(self.model.predict(features))
        predictions = self.model.predict(features)
        return predictions.flatten()
        


    