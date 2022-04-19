from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from rainflow import extract_cycles

from pyStruct.machines.feature_processors import *
from pyStruct.machines.optimizers import *
from pyStruct.machines.regressors import *
from pyStruct.machines.structures import *
from pyStruct.machines.reconstructors import *


FEATURE_PROCESSORS = {
    'COHERENT_STRENGTH': PodCoherentStrength
}

OPTIMIZERS = {
    'POSITIVE_WEIGHTS': PositiveWeights,
    'ALL_WEIGHTS':AllWeights,
    'INTERCEPT': InterceptAndWeights,
    'POS_INTERCEPT': PositiveInterceptAndWeights,
}

WEIGHTS_PREDICTORS ={
    'BAYESIAN': BayesianModel,
    'MULTIBAYESIAN':MultiLevelBayesian,
    'GB': GbRegressor,
    'BSPLINE':BSpline,
    'NN': NnRegressor
}
STRUCTURE_PREDICTORS = {
    'GB': GBLookupStructure,
    'BAYESIAN': BayesianLookupStructure
}
RECONSTRUCTORS = {
    'LINEAR': LinearReconstruction,
}


class TwoMachineFramework:
    def __init__(self, config):
        # initialize the machines
        self.config = config

        # Feature processor 
        self.feature_processor = FEATURE_PROCESSORS[config.machines.feature_processor.upper()](config.features)
        self.structure_library = self.feature_processor.get_structure_tables()

        # Structure predictor
        self.structure_predictor = STRUCTURE_PREDICTORS[config.machines.structure_predictor.upper()](config.structures, self.structure_library)

        # Optimization
        self.optimizer = OPTIMIZERS[config.machines.optimizer.upper()]()

        # Predict weights
        self.weights_predictor = WEIGHTS_PREDICTORS[config.machines.weights_predictor.upper()]

        # Reconstructor
        self.reconstructor = RECONSTRUCTORS[config.machines.reconstructor.upper()]()
        
    def _train_structure(self):
        # Specify save folder
        to_folder = Path(self.config.paths.save_to)/'structure_predictor'
        to_folder.mkdir(parents=True, exist_ok=True)
        self.structure_predictor.set_model_path(to_folder)
        
        if self.structure_predictor.save_to.exists():
            self.structure_predictor.load()
        else:
            # get table and train
            self.structure_predictor.train()
            self.structure_predictor.save()
        return
    
    def _validate_structure(self):
        # Specify save folder
        to_folder = Path(self.config.paths.save_to)/'structure_predictor'
        to_folder.mkdir(parents=True, exist_ok=True)
        self.structure_predictor.set_model_path(to_folder)

        # load model
        self.structure_predictor.load()

        training_bcs = [sample.bc for sample in self.feature_processor.training_samples]
        for bc in training_bcs:
            print(f'\n-----training bc: {bc}----\n')
            predictions = self.structure_predictor.predict(bc)
            for mode in range(len(predictions)):
                print(f'pred, mode {mode}: {predictions[mode].bc}')
    
    def _optimize(self):
        # Specify folder 
        to_folder = Path(self.config.paths.save_to)/'optimization'
        to_folder.mkdir(parents=True, exist_ok=True)
        records = pd.DataFrame()

        # Decoupled method: this part is independent of the structure prediction
        # compose temporal matrix
        # by training samples
        samples = self.feature_processor.training_samples
        for sample in samples:
            for theta_deg in self.config.features.theta_degs:
                # Construct X, y 
                X = sample.pod.X_temporal
                y = sample.walls[f'{theta_deg:.2f}'].temperature
                # Optimize
                weights = self.optimizer.optimize(X, y)
                # Keep record
                record= {'sample':sample.name, 'theta_deg':theta_deg}
                for mode, w in enumerate(weights):
                    record[f'w{mode}'] = w
                records = pd.concat([records, pd.DataFrame(record, index=[0])], ignore_index=True)
        
        records.to_csv(to_folder/'optimization.csv', index=False)
        return

    def validate_optimize(self) -> list[Sample]:
        # Read table
        to_folder = Path(self.config.paths.save_to)/'optimization'
        records = pd.read_csv(to_folder/'optimization.csv')
        # TODO: Error check
        samples = self.feature_processor.training_samples
        for sample in samples:
            for theta_deg in self.config.features.theta_degs:
                record = records[(records['sample'] == sample.name) & (records['theta_deg'] == theta_deg)]
                weights = record[[f'w{mode}' for mode in range(self.config.features.N_modes)]].values
                X = sample.pod.X_temporal
                y_optm = self.reconstructor.reconstruct(X, weights)
                sample.record_optimize(theta_deg, weights, y_optm)
        return samples
        
    def train(self):
        # Train Structure
        self._train_structure()

        # Valiation: 
        self._validate_structure()
        # compare structure_targets and structure_predictions

        # Train Weights
        # optimize 
        self._optimize()

        # weights_inputs = structure_features
        # weights_predictions = self.weights_predictor.train(weights_inputs, weights_targets)

        # todo: valiation: 
        # compare weights_targets and weights_predictions
        #
        return

    def predict(self):
        pass


class TwoMachineFramework_depr:
    def __init__(
        self,
        config: dict,
        feature_processor,
        optimizer, 
        weights_predictor, 
        structure_predictor,
        start_new
    ):
        # Assertions: Make sure each inputs are of the right class
        self.config = config
        self.feature_processor = feature_processor(config)
        self.optimizer = optimizer(config)
        self.weights_predictor = weights_predictor(config)
        self.structure_predictor = structure_predictor(config)

        # save the feature table
        self.save_to = Path(self.config['save_to'])
        self.feature_table = self.feature_processor.feature_table
        self.feature_table.to_csv(self.save_to/'feature_table.csv', index=None)

        # save the config
        if start_new:
            shutil.rmtree(self.save_to, ignore_errors=True)
        self.save_to.mkdir(parents=True, exist_ok=True)
        config_file = self.save_to/'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f)

        # check file
        self._optm_feature_table_file = self.save_to / 'optm_feature_table.csv'


    def train(self, show=False):
        # Weight optimization 
        print(f'=========== Optimization ==========')
        if not self._optm_feature_table_file.exists():
            self._optimize()
            self._move_neg_sign_to_features()
        else:
            self._read_optm_feature_table()

        # Time-series Predictor: the models are saved in the weigths_predictors
        # check weights_predictors
        print(f'=========== Weights ==========')
        if not self.weights_predictor.save_as.exists():
            self.weights_predictor.train(self.feature_table)
        else:
            self.weights_predictor.load(self.feature_table)

        # Structure Predictor
        print(f'=========== Structures ==========')
        if not self.structure_predictor.save_as.exists():
            self.structure_predictor.train(self.feature_table)
        else:
            self.structure_predictor.load(self.feature_table)

    def _reconstruct(self, X, weights, T_0=None):
        # Intercept model 
        if not T_0:
            print("No T_0 is provide; only predict the fluc")
            T_0 =0
        y_pred = T_0+ np.matmul(weights, X)
        return y_pred

    def predict(
        self, 
        inputs: dict, 
        perturb_structure:bool = False, 
        perturb_weights=False, 
        weights = None):

        assert list(inputs.keys())  == self.config['x_labels']

        # Retrieve base X: 
        X, _ = self.feature_processor.get_training_pairs()

        # Predict structures
        X_compose, features  = self.structure_predictor.predict_and_compose(inputs, X, perturb_structure)

        # Predict weights
        if not weights:
            if perturb_weights:
                weights = self.weights_predictor.sampling(features, N_realizations=1)
            else:
                weights = self.weights_predictor.predict(features)       
        
        y_pred = self._reconstruct(X_compose, weights)
        return y_pred


    def _create_empty_level_table(self, theta_deg):
        # Initialize level tabel
        level_table = self.feature_processor.pods.bcs.copy()
        level_table['theta_deg'] = np.ones(len(level_table)) * theta_deg
        level_table['T_0'] = np.zeros(len(level_table))
        return level_table

    def _optimize(self):
        """ For every theta_deg, every sample, do optimization"""
        # Initalize a place holder 
        self.feature_table['w'] = np.zeros(len(self.feature_table))
        self.level_table = pd.concat([ self._create_empty_level_table(theta_deg) for theta_deg in self.config['theta_degs']], ignore_index=True)

        # Run optimization and Create table 
        for theta_deg, sample in itertools.product(self.config['theta_degs'], np.arange(self.config['N_modes'])):
            print(f"Optimizing Theta= {theta_deg}, Sample={sample}")

            # Retreve data
            X, y = self.feature_processor.get_training_pairs(theta_deg)
            X_r = X[sample, ...]
            y_r = y[sample, ...]

            # Optimize 
            filtered_feature = self.feature_table[ (self.feature_table['sample']==sample) & (self.feature_table['theta_deg']== theta_deg) ]
            id = filtered_feature.index

            # Call the optimizer
            weights = self.optimizer.optimize(X_r, y_r)

            # Keep record
            if len(weights) == self.config['N_modes']:
                for i, w in enumerate(weights):
                    self.feature_table.loc[id[i], 'w'] = w
            else:
                # keep record in level table
                self.level_table.loc[(self.level_table['sample']==sample) & (self.level_table['theta_deg']== theta_deg), 'T_0'] = weights[0]
                # keep record in feature table
                for i, w in enumerate(weights[1:]):
                    self.feature_table.loc[id[i], 'w'] = w

        # Save the result
        self.level_table.to_csv(self.save_to/'level_table.csv', index=None)
        self.feature_table.to_csv(self._optm_feature_table_file, index=None)
        return

    def _read_optm_feature_table(self):
        print("read feature table")
        self.feature_table = pd.read_csv(self._optm_feature_table_file)
        self.feature_processor.feature_table = self.feature_table
        self.level_table = pd.read_csv(self.save_to/'level_table.csv')
        return
    
