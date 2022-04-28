import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rainflows import extract_cycles

from framework import Machine

def get_rainflow(signal):
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
def get_signal_descriptions(signal: np.ndarray):
    # des ={
    #     'mean':signal.mean(),
    #     'std':signal.std(),
    #     'counts':get_rainflow(signal)['counts']
    # }
    return signal.mean(), signal.std(), get_rainflow(signal)['counts']

class Validation:
    """ See the performance of a trained machine"""
    def __init__(self, machine: Machine):
        pass

    def optimizations(self):
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

    def structure_predictions(self):
        pass

    def weights_predictions(self):
        pass

    def full_predictions(self):
        """ Visualize the training samples """
        # Plot setting
        ncols=5
        nrows = int(np.ceil(self.config['N_samples']/ncols))
        figsize = (ncols*3.2, nrows*3)
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=figsize)
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=figsize)

        X, y_trues = self.feature_processor.get_training_pairs(theta_deg)
        ft = self.feature_table[self.feature_table['theta_deg'] == theta_deg]

        for sample in range(self.config['N_samples']):
            ranked_feature_by_sample = ft[ft['sample'] ==sample].sort_values(by=['rank'])
            ax1 = axes1[sample//ncols, sample%ncols]
            ax2 = axes2[sample//ncols, sample%ncols]

            # Plot optimize
            w_optm = ranked_feature_by_sample['w'].values
            y_optm = self._reconstruct(X[sample, ...], w_optm)

            # Plot predict
            inputs = {x_label: ranked_feature_by_sample[x_label].values[0] for x_label in self.config['x_labels']}
            X_compose, features  = self.structure_predictor.predict_and_compose(inputs, X, perturb_structure=False)
            w_pred = self.weights_predictor.predict(features)

            T_0 = self.level_table[(self.level_table['sample']==sample) & (self.level_table['theta_deg']==theta_deg)]['T_0'].iloc[0]
            y_pred = self._reconstruct(X_compose, w_pred, T_0)

            # Plot: 
            ax1.plot(y_trues[sample, :], label='True')
            ax1.plot(y_pred, label='Pred')

            # Plot weights prediction
            ax2.stem(w_optm)
            ax2.stem(w_pred, markerfmt='r*')

        plt.legend(bbox_to_anchor=(0, 1.1))
        plt.tight_layout()
        fig1.savefig(self.save_to/f'validation_deg{theta_deg}_signal.png')
        fig2.savefig(self.save_to/f'validation_deg{theta_deg}_weights.png')
        plt.show()
        return 

    def rainflows(self):
        X, y_trues = self.feature_processor.get_training_pairs(theta_deg)
        ft = self.feature_table[self.feature_table['theta_deg'] == theta_deg]

        col_names = ['sample', 'true_mean', 'pred_mean','true_std', 'pred_std', 'true_counts', 'pred_counts']
        comp_array = np.zeros((self.config['N_samples'], len(col_names)))

        for sample in range(self.config['N_samples']):
            ranked_feature_by_sample = ft[ft['sample'] ==sample].sort_values(by=['rank'])

            # Plot optimize
            w_optm = ranked_feature_by_sample['w'].values
            y_optm = self._reconstruct(X[sample, ...], w_optm)

            # Plot predict
            inputs = {x_label: ranked_feature_by_sample[x_label].values[0] for x_label in self.config['x_labels']}
            X_compose, features  = self.structure_predictor.predict_and_compose(inputs, X, perturb_structure=False)
            w_pred = self.weights_predictor.predict(features)

            T_0 = self.level_table[(self.level_table['sample']==sample) & (self.level_table['theta_deg']==theta_deg)]['T_0'].iloc[0]
            y_pred = self._reconstruct(X_compose, w_pred, T_0)
            
            # Get descriptions
            true_mean, true_std, true_counts = get_signal_descriptions(y_trues[sample,...])
            pred_mean, pred_std, pred_counts = get_signal_descriptions(y_pred)

            comp_array[sample, :] = (sample, true_mean, pred_mean, true_std, pred_std, true_counts, pred_counts)

        df = pd.DataFrame(comp_array, columns=col_names)

            # Plot: 
        return df
