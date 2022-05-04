from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pyStruct.optimizers.optimizer import Optimizer
from pyStruct.optimizers.losses import *


def intercept_objective(w, X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_weight, initial_weighting):
    N_modes, _ = X_r.shape
    e_fft_0, e_std_0, e_min_0, e_max_0, e_hist_0 = initial_weighting 
    y_pred = w[-1] + np.matmul(w[:N_modes], X_r)
    e_fft = fft_loss(y_r, y_pred)
    e_std = std_loss(y_r, y_pred)
    e_min = min_loss(y_r, y_pred)
    e_max = max_loss(y_r, y_pred)
    e_hist = hist_loss(y_r, y_pred)

    error = fft_loss_weight*e_fft/e_fft_0 + std_loss_weight * e_std/e_std_0 + min_loss_weight* e_min/e_min_0 + max_loss_weight* e_max/e_max_0 + hist_weight* e_hist/e_hist_0
    return error

class InterceptRealWeights(Optimizer):
    def __init__(self, folder: Path, loss_weights_config=None):
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
        self.save_to = folder 
        self.all_weights = None

    def get_constraint(self, x):
        return tuple([{'type':'ineq', 'fun': lambda x,i=i: x[i] - x[i+1]} for i in range(1, len(x)-1)] )

    def set_all_weights(self, all_weights: np.ndarray):
        self.all_weights = all_weights

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
        w_init = np.arange(N_modes+1)[::-1]*0.1
        # constraint = self.get_constraint(w_init)
        result = minimize(
            intercept_objective,
            w_init, 
            method='SLSQP',
            args=(X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight, initial_weightings),
            bounds = ((-1, 1) for i in range( N_modes+1)),
            # constraints=constraint, 
            options={'maxiter': max_iter}
            )
        return result.x


class InterceptPositive(InterceptRealWeights):
    def __init__(self, loss_weights_config=None):
        super(InterceptPositive, self).__init__(loss_weights_config)

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
        w_init = np.arange(N_modes+1)[::-1]*0.1
        # constraint = self.get_constraint(w_init)
        result = minimize(
            intercept_objective,
            w_init, 
            method='SLSQP',
            args=(X_r, y_r, fft_loss_weight, std_loss_weight, min_loss_weight, max_loss_weight, hist_loss_weight, initial_weightings),
            bounds = ((0, np.inf) for i in range( N_modes+1)),
            # constraints=constraint, 
            options={'maxiter': max_iter}
            )
        return result.x