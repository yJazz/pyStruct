from pathlib import Path
import json
import numpy as np
import pandas as pd
import wandb
from scipy.optimize import minimize
from typing import Protocol
from scipy.fft import fft, fftfreq

def fft_loss(y, y_pred):
    fft_y = fft(y)
    fft_y_pred = fft(y_pred)
    
    # Find the distance between the two vectors 
    loss = np.linalg.norm(fft_y - fft_y_pred)
    return loss

def std_loss(y, y_pred):
    std_true = y.std()
    std_pred = y_pred.std()
    return np.abs(std_true-std_pred)

def max_loss(y, y_pred):
    true_max = y.max()
    pred_max = y_pred.max()
    error = np.abs(true_max - pred_max)
    return error

def min_loss(y, y_pred):
    true_min = y.min()
    pred_min = y_pred.min()
    error = np.abs(true_min - pred_min)
    return error

def hist_loss(y, y_pred):
    bins = np.linspace(-0.1, 0.1, 29)
    h, b = np.histogram(y, bins=bins)
    h_pred, b = np.histogram(y_pred, bins=bins)
    return np.linalg.norm(h-h_pred)


class Optimizer(Protocol):
    def init_config(self):
        pass
    def optimize(self):
        pass

class AllWeights:
    def __init__(self, config, loss_weights_config=None):
        self.config = config
        if loss_weights_config:
            assert len(loss_weights_config) == 5, "Need to specify: fft, std, min, max, hist"

        if not loss_weights_config:
            loss_weights_config = [10, 1, 1, 1, 1]

        self.config ={
            "fft_loss_weight":loss_weights_config[0],
            "std_loss_weight": loss_weights_config[1],
            "min_loss_weight":loss_weights_config[2],
            "max_loss_weight":loss_weights_config[3], 
            "hist_loss_weight":loss_weights_config[4],
            "maxiter":1000,
        }

    def optimize(
        self,
        X_r, 
        y_r, 
        ):

        fft_loss_weight=self.config['fft_loss_weight']
        std_loss_weight = self.config['std_loss_weight']
        min_loss_weight = self.config['min_loss_weight']
        max_loss_weight = self.config['max_loss_weight']
        hist_loss_weight = self.config['hist_loss_weight']
        max_iter=self.config['maxiter']

        N_modes, N_t = X_r.shape
        initial_weightings = initialize_loss_weighting(X_r, y_r)
        w_init = np.arange(N_modes)*0.1
        # constraint = self.get_constraint(N_modes)
        constraint = self.get_constraint(w_init)
        bounds = [(-1, 1) for i in range(N_modes)]
        result = minimize(
            objective_function, 
            w_init, 
            method='SLSQP',
            args=(X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight, initial_weightings),
            # constraints=constraint,  
            bounds=bounds,
            options={'maxiter': max_iter}
            )
        return result.x
        
    # def get_constraint(self, N_modes):
    #     cons = []
    #     for i in range(N_modes-1):
    #         con = lambda x: abs(x[i+1]) - abs(x[i])
    #         cons.append({'type': 'ineq', 'fun': con})
    #     return cons 
    def get_constraint(self, x):
        return tuple([{'type':'ineq', 'fun': lambda x,i=i: abs(x[i]) - abs(x[i+1])} for i in range(len(x)-1)] )
 
class PositiveWeights:
    def __init__(self, config, loss_weights_config=None):
        self.config = config
        if loss_weights_config:
            assert len(loss_weights_config) == 5, "Need to specify: fft, std, min, max, hist"

        if not loss_weights_config:
            loss_weights_config = [10, 1, 1, 1, 1]

        self.config ={
            "fft_loss_weight":loss_weights_config[0],
            "std_loss_weight": loss_weights_config[1],
            "min_loss_weight":loss_weights_config[2],
            "max_loss_weight":loss_weights_config[3], 
            "hist_loss_weight":loss_weights_config[4],
            "maxiter":1000,
        }
    def get_constraint(self, x):
        return tuple([{'type':'ineq', 'fun': lambda x,i=i: x[i] - x[i+1]} for i in range(len(x)-1)] )

    def optimize(
        self,
        X_r, 
        y_r, 
        ):

        fft_loss_weight=self.config['fft_loss_weight']
        std_loss_weight = self.config['std_loss_weight']
        min_loss_weight = self.config['min_loss_weight']
        max_loss_weight = self.config['max_loss_weight']
        hist_loss_weight = self.config['hist_loss_weight']
        max_iter=self.config['maxiter']

        N_modes, N_t = X_r.shape
        initial_weightings = initialize_loss_weighting(X_r, y_r)
        w_init = np.arange(N_modes)[::-1]*0.1
        constraint = self.get_constraint(w_init)
        result = minimize(
            objective_function, 
            w_init, 
            method='SLSQP',
            args=(X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight, initial_weightings),
            bounds = ((0, 1) for i in range(N_modes)),
            constraints=constraint, 
            options={'maxiter': max_iter}
            )
        return result.x


def initialize_loss_weighting(X_r, y_r):
    N_modes = X_r.shape[0]
    w = np.ones(N_modes) * 0.5
    y_pred = np.matmul(w, X_r)
    e_fft_0 = fft_loss(y_r, y_pred)
    e_std_0 = std_loss(y_r, y_pred)
    e_min_0 = min_loss(y_r, y_pred)
    e_max_0 = max_loss(y_r, y_pred)
    e_hist_0 = hist_loss(y_r, y_pred)
    return e_fft_0, e_std_0, e_min_0, e_max_0, e_hist_0


def objective_function(w, X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_weight, initial_weighting):
    e_fft_0, e_std_0, e_min_0, e_max_0, e_hist_0 = initial_weighting 
    y_pred = np.matmul(w, X_r)
    e_fft = fft_loss(y_r, y_pred)
    e_std = std_loss(y_r, y_pred)
    e_min = min_loss(y_r, y_pred)
    e_max = max_loss(y_r, y_pred)
    e_hist = hist_loss(y_r, y_pred)

    # print(f"fft_loss: {e_fft}, std_loss: {e_std}, min_loss: {e_min}, max_loss: {e_max}, hist_loss:{e_hist}")
    error = fft_loss_weight*e_fft/e_fft_0 + std_loss_weight * e_std/e_std_0 + min_loss_weight* e_min/e_min_0 + max_loss_weight* e_max/e_max_0 + hist_weight* e_hist/e_hist_0
    # print(f"error: {error}")
    # wandb.log({
    #     'fft_loss': e_fft, 
    #     'std_loss': e_std, 
    #     'min_loss': e_min,
    #     'max_loss': e_max,
    #     'hist_loss': e_hist,
    #     'weighted_loss': error
    # })
    return error

def get_constraint(x):
    cons = []
    for i in range(len(x)-1):
        con = lambda x: x[i] > x[i+1]
        # con = lambda x: x[i]
        cons.append({'type': 'ineq', 'fun': con})
    return cons
def get_constraint(x):
    return tuple([{'type':'ineq', 'fun': lambda x,i=i: x[i] - x[i+1]} for i in range(len(x)-1)] )

def optimize(X_r, y_r, maxiter, 
             fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight):
    N_modes, N_t = X_r.shape
    # bounds = [(-1, 1) for i in range(N_modes)]
    initial_weightings = initialize_loss_weighting(X_r, y_r)
    # w_init = np.random.random(size=(N_modes,1)) * 1
    w_init = np.arange(N_modes)*0.1
    constraint = get_constraint(w_init)
    result = minimize(
        objective_function, 
        w_init, 
        method='SLSQP',
        args=(X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight, initial_weightings),
        # bounds= bounds, 
        bounds = ((0, None) for i in range(N_modes)),
        constraints=constraint,  
        options={'maxiter': maxiter})
    return result.x



def reconstruct_optimize(config: dict):
    X, y = get_training_pairs(
        config['workspace'], 
        theta_deg=config['theta_deg'],
        N_t = config['N_t']
        )
    weights_table = config['weights_table']

    N_weights = len(config['mode_ids'])
    y_preds = []
    for i in config['sample_ids']:
        weights = weights_table.iloc[i, -N_weights:].values

        X_r = X[i, config['mode_ids'], :]
        y_r = y[i, ...]
        y_pred = np.matmul(weights, X_r)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)
    return y[config['sample_ids'], ...], y_preds




