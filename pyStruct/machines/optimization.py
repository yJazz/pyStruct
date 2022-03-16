from pathlib import Path
import json
import numpy as np
import pandas as pd
import wandb
from scipy.optimize import minimize

from pyStruct.machines.loss import fft_loss, std_loss, min_loss, max_loss, hist_loss

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


def optm_workflow(config, X, y):
    # Get X,y pair

    # Read statepoint
    with open(Path(config['workspace'])/"signac_statepoint.json") as f:
        sp = json.load(f)
        aggre_sp = sp['aggre_sp']

    # Run optimization and Create table 
    weights_df = pd.DataFrame()
    for i in config['sample_ids']:
        proc_id = f'proc_{i}'
        print(f"Optimizing case {i}")
        # Get modes in single case
        X_r = X[i, range(config['N_modes']), :]
        y_r = y[i, ...]


        fft_loss_weight=config['fft_loss_weight']
        std_loss_weight = config['std_loss_weight']
        min_loss_weight = config['min_loss_weight']
        max_loss_weight = config['max_loss_weight']
        hist_loss_weight = config['hist_loss_weight']
        max_iter=config['maxiter']

        weights = optimize(X_r, y_r, maxiter=max_iter,
                            fft_loss_weight=fft_loss_weight, 
                           std_loss_weight=std_loss_weight, min_loss_weight=min_loss_weight, max_loss_weight=max_loss_weight, hist_loss_weight=hist_loss_weight)
        
        # Record boundary conditions and weights
        record = {}
        record['sample']=i
        bc = aggre_sp[proc_id]['cfd_sp']['bc']
        for key in bc.keys():
            record[key] = bc[key]
        for mode in range(config['N_modes']):
            record[f'w{mode}'] = weights[mode]
        record = pd.DataFrame(record, columns=record.keys(), index=[0])

        weights_df = pd.concat([weights_df, record], axis=0, ignore_index=True)
    
    return weights_df

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




