"""
This module contains object for verifying/validating the prediction

"""
from pathlib import Path
from pyStruct.machines.framework import TimeSeriesPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class VisualizeOptimization:
    def __init__(self, config, ncols=5):
        self.config = config
        self.wp = self._init_data(config)

        # plot setting
        self.ncols=5
        self.N_samples = config['N_samples']
        self.nrows = ceil(self.N_samples/self.ncols)

    def _init_data(self, config):
        wp_path = config['save_to']/'ts_regression'/'wp.csv'
        wp = pd.read_csv(wp_path)
        return wp

    def read_weights_from_table(self, theta_deg, samples):
        wp = self.wp
        weights ={}
        for sample in samples:
            weights[sample] = []
            wp_sub = wp[(wp['theta_deg']==theta_deg) & (wp['sample']==sample)]
            for mode in range(self.config['N_modes']):
                w = wp_sub[wp_sub['mode']==mode].w.iloc[0]
                weights[sample].append(w)
        
        return weights


    
    def plot_T_probe(self, theta_deg, weights, figsize=(16, 12)):
        # get xy
        self.config['theta_deg'] = theta_deg
        ts = TimeSeriesPredictor(
            N_samples=self.config['N_samples'], 
            N_modes=self.config['N_modes'], 
            N_t=self.config['N_t'], 
            wp=self.wp
        )
        wp = self.wp
        X, y, idx = ts.get_training_pairs(self.config)

        # Plot 
        fig_1, axes_1 = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize)
        fig_2, axes_2 = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize)


        for sample in weights.keys():
            ax_1 = axes_1[sample//self.ncols, sample%self.ncols]
            ax_2 = axes_2[sample//self.ncols, sample%self.ncols]

            # predict 
            ax_1.set_title(sample)
            y_pred = np.array(
                    [X[sample, mode, :] * weights[sample][mode] for mode in range(self.config['N_modes']) ] ).sum(axis=0)
            ax_1.plot(y[sample][-self.config['N_t']:])
            ax_1.plot(y_pred)

            # Ax2 : Stem
            all_weights = np.zeros(20)
            for id, w in zip(idx[sample], weights[sample]):
                all_weights[id] = w
            ax_2
            ax_2.stem(all_weights)
            ax_2.set_xticks(np.arange(len(idx[sample])))
            ax_2.set_xticklabels(idx[sample])
        plt.show()

        fig_1.savefig(self.config['save_to']/'figures'/f'optm_pred_theta_{theta_deg}.png')
        fig_2.savefig(self.config['save_to']/'figures'/f'optm_w_theta_{theta_deg}.png')
