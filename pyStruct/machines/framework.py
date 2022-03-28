from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class TwoMachineFramework:
    def __init__(
        self,
        config: dict,
        feature_processor,
        optimizer, 
        weights_predictor, 
        structure_predictor
    ):
        # Assertions: Make sure each inputs are of the right class
        self.config = config
        self.feature_processor = feature_processor(config)
        self._init_data()
        self.optimizer = optimizer(config)
        self.weights_predictor = weights_predictor(config)
        self.structure_predictor = structure_predictor(config)

    def _init_data(self):
        self.feature_table = self.feature_processor.feature_table

        # save the config
        self.save_to = Path(self.config['save_to'])
        self.save_to.mkdir(parents=True, exist_ok=True)
        config_file = self.save_to/'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
    
        # check file
        self._optm_feature_table_file = self.save_to / 'feature_table.csv'

    def train(self, show=False):
        # Weight optimization 
        print(f'=========== Optimization ==========')
        if not self._optm_feature_table_file.exists():
            self._optimize()
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

    
    def predict(self, inputs: dict, perturb_structure:bool = False, perturb_weights=False, weights = None):
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
        print(weights)
        
        y_pred = self._reconstruct(X_compose, weights)
        return y_pred

    def _reconstruct(self, X, weights):
        # Output
        y_pred = np.array(
            [X[mode, :] * weights[mode] for mode in range(self.config['N_modes']) ] ).sum(axis=0)
        return y_pred

    def prediction_validation(self):
        """ Visualize the training samples """
        # Get reference data
        X, y_trues = self.feature_processor.get_training_pairs()
        y_preds = []

        # Get boundary conditions
        feature_table = self.feature_table
        # for sample in range(self.config['N_samples']):
        ncols=5
        nrows = int(np.ceil(self.config['N_samples']/ncols))
        figsize = (ncols*3.2, nrows*3)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for sample in range(20):
            ax = axes[sample//ncols, sample%ncols]
            # Optimize
            feature_by_sample = self.feature_table[self.feature_table['sample']==sample]
            feature_by_sample = feature_by_sample.sort_values(by='rank')
            weights = feature_by_sample['w'].values
            y_optm = self._reconstruct(X[sample, ...], weights)

            # Predict
            f = feature_table[feature_table['sample'] == sample]
            input = {x_label: f[x_label].values[0] for x_label in self.config['x_labels']}
            y_pred = self.predict(input)
            y_preds.append(y_pred)

            # Plot
            ax = axes[sample//ncols, sample%ncols]
            ax.plot(y_trues[sample, :], label='True')
            ax.plot(y_optm, label = 'Optm')
            ax.plot(y_preds[sample], label='Pred')

        plt.legend(bbox_to_anchor=(0, 1.1))
        plt.tight_layout()
        plt.savefig(self.save_to/'validation.png')
        plt.show()
        return y_trues, y_preds

    def optimization_validation(self):
        X, y_trues = self.feature_processor.get_training_pairs()
        
        ncols = 5
        nrows = int(np.ceil(self.config['N_samples']/ncols))
        figsize = (ncols*3.2, nrows*3)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for sample in range(self.config['N_samples']):
            feature_by_sample = self.feature_table[self.feature_table['sample']==sample]
            feature_by_sample = feature_by_sample.sort_values(by='rank')
            input = {x_label: feature_by_sample[x_label].values[0] for x_label in self.config['x_labels']}
            y_pred = self.predict(input)

            ax = axes[sample//ncols, sample%ncols]
            ax.plot(y_trues[sample, :], label='True')
            
            # optimization
            weights = feature_by_sample['w'].values
            y_optm = self._reconstruct(X[sample, ...], weights)
            ax.plot(y_optm, label='Optm')

        plt.legend(bbox_to_anchor=(0, 1.1))
        plt.tight_layout()
        plt.savefig(self.save_to/'optimization.png')
        plt.show()

    def _optimize(self):
        # Retrieve data
        X, y = self.feature_processor.get_training_pairs()

        # Create Space holder
        self.feature_table['w'] = np.zeros(len(self.feature_table))

        # Run optimization and Create table 
        for sample in range(self.config['N_samples']):
            feature_by_sample = self.feature_table[self.feature_table['sample']==sample]
            feature_by_sample = feature_by_sample.sort_values(by='rank')
            id = feature_by_sample.index

            print(f"Optimizing case {sample}")
            proc_id = f'proc_{sample}'

            # Get modes in single case
            X_r = X[sample, range(self.config['N_modes']), :]
            y_r = y[sample, ...]

            # Call the optimizer
            weights = self.optimizer.optimize(X_r, y_r)

            for i, w in enumerate(weights):
                self.feature_table.loc[id[i], 'w'] = w
        # Save the result
        self.feature_table.to_csv(self._optm_feature_table_file, index=None)
        return

    def _read_optm_feature_table(self):
        print("read feature table")
        self.feature_table = pd.read_csv(self._optm_feature_table_file)
        self.feature_processor.feature_table = self.feature_table
        return