from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rainflow import extract_cycles


class TwoMachineFramework:
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

    def _reconstruct(self, X, weights, T_0):
        # Intercept model 
        y_pred = T_0+ np.matmul(weights, X)
        return y_pred


    def prediction_validation(self):
        """ Visualize the training samples """

        # Plot setting
        ncols=5
        nrows = int(np.ceil(self.config['N_samples']/ncols))
        figsize = (ncols*3.2, nrows*3)
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=figsize)
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=figsize)

        for sample in range(self.config['N_samples']):
            ranked_feature_by_sample = self.feature_table[self.feature_table['sample']==sample].sort_values(by=['rank'])
            ax1 = axes1[sample//ncols, sample%ncols]
            ax2 = axes2[sample//ncols, sample%ncols]

            # Plot optimize
            w_optm = ranked_feature_by_sample['w'].values
            y_optm = self._reconstruct(X[sample, ...], w_optm)

            # Plot predict
            inputs = {x_label: ranked_feature_by_sample[x_label].values[0] for x_label in self.config['x_labels']}
            X_compose, features  = self.structure_predictor.predict_and_compose(inputs, X, perturb_structure=False)
            w_pred = self.weights_predictor.predict(features)
            y_pred = self._reconstruct(X_compose, w_pred)

            # Plot: 
            ax1.plot(y_trues[sample, :], label='True')
            ax1.plot(y_pred, label='Pred')

            # Plot weights prediction
            ax2.stem(w_optm)
            ax2.stem(w_pred, markerfmt='r*')

        plt.legend(bbox_to_anchor=(0, 1.1))
        plt.tight_layout()
        fig1.savefig(self.save_to/'validation.png')
        plt.show()
        return 

    def optimization_validation(self):
        for theta_deg in self.config['theta_degs']:
            feature_theta = self.feature_table[self.feature_table['theta_deg'] == theta_deg]
            ncols = 5
            nrows = int(np.ceil(self.config['N_samples']/ncols))
            figsize = (ncols*3.2, nrows*3)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

            X, y_trues = self.feature_processor.get_training_pairs(theta_deg)

            for sample in range(self.config['N_samples']):
                ranked_feature_by_sample = feature_theta[feature_theta['sample'] == sample].sort_values(by=['rank'])

                ax = axes[sample//ncols, sample%ncols]
                ax.plot(y_trues[sample, :], label='True')
                
                # optimization
                weights = ranked_feature_by_sample['w'].values
                T_0 = self.level_table[(self.level_table['sample']==sample) & (self.level_table['theta_deg']==theta_deg)]['T_0'].iloc[0]
                y_optm = self._reconstruct(X[sample, ...], weights, T_0)
                ax.plot(y_optm, label='Optm')
            
            plt.suptitle(f'theta={theta_deg}')
            plt.legend(bbox_to_anchor=(0, 1.1))
            plt.tight_layout()
            plt.savefig(self.save_to/'optimization.png')
            plt.show()

    def _optimize(self):
        # Initalize a place holder 
        self.feature_table['w'] = np.zeros(len(self.feature_table))
        level_tables=[]

        # Run optimization and Create table 
        for theta_deg in self.config['theta_degs']:
            print(f"Optimizing Theta= {theta_deg}")

            # Initialize level tabel
            level_table = self.feature_processor.pods.bcs.copy()
            level_table['theta_deg'] = np.ones(len(level_table)) * theta_deg
            level_table['T_0'] = np.zeros(len(level_table))

            # Retrieve data
            X, y = self.feature_processor.get_training_pairs(theta_deg)
            feature_theta = self.feature_table[self.feature_table['theta_deg'] == theta_deg]

            for sample in range(self.config['N_samples']):
                print(f"- optimizing case {sample}")
                proc_id = f'-- proc_{sample}'
                feature_by_sample = feature_theta[feature_theta['sample'] == sample]
                id = feature_by_sample.index

                # Get modes in single case
                X_r = X[sample, range(self.config['N_modes']), :]
                y_r = y[sample, ...]

                # Call the optimizer
                weights = self.optimizer.optimize(X_r, y_r)
                if len(weights) == self.config['N_modes']:
                    for i, w in enumerate(weights):
                        self.feature_table.loc[id[i], 'w'] = w
                else:
                    level_table.loc[level_table['sample']==sample, 'T_0'] = weights[0]
                    for i, w in enumerate(weights[1:]):
                        self.feature_table.loc[id[i], 'w'] = w
            level_tables.append(level_table)

        # Compose all the level table
        self.level_table = pd.concat(level_tables, ignore_index=True)
        self.level_table.to_csv(self.save_to/'level_table.csv', index=None)

        # Save the result
        self.feature_table.to_csv(self._optm_feature_table_file, index=None)
        return

    def _read_optm_feature_table(self):
        print("read feature table")
        self.feature_table = pd.read_csv(self._optm_feature_table_file)
        self.feature_processor.feature_table = self.feature_table
        self.level_table = pd.read_csv(self.save_to/'level_table.csv')
        return

    def get_rainflow(self, signal):
        rf = []
        for rng, mean, count, i_start, i_end in extract_cycles(signal): 
            rf.append((rng, mean, count, i_start, i_end))
        rf = np.array(rf)
        output={
            'range':rf[:, 0],
            'mean':rf[:, 1],
            'N_i':rf[:,2],
            'i_start':rf[:, 3],
            'i_end':rf[:,4],
            'counts': sum(rf[:, 2])
        }
        return output
