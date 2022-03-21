from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import arviz as az 
import pymc3 as pm
import scipy.stats as stats
from theano import shared
from sklearn.preprocessing import StandardScaler

from pyStruct.machines.validation import VisualizeOptimization

class BayesMultiDimModel:
    def __init__(self, name:str, wp: pd.DataFrame, var_names: list[str], target_name:str, save_to):
        self.name = name
        self.wp = wp
        self.var_names = var_names
        self.target_name = target_name
        self.save_to = Path(save_to)/f"bayesian_model_{name}"
        self.save_to.mkdir(parents=True, exist_ok=True)

        # get data
        self.target = self.wp[target_name].values
        self.inputs = {}
        self.inputs_shared={}
        self.standrdizers = {}
        for var_name in var_names:
            input, standrdizer = self.transform_input(var_name)
            self.inputs_shared[var_name] = shared(input) # the shared tensor, which can be modified later
            self.standrdizers[var_name] = standrdizer
            self.inputs[var_name] = input # this won't change

    def transform_input(self, var_name):
        standardizer = StandardScaler()
        var = standardizer.fit_transform(self.wp[var_name].values.reshape(-1,1)).flatten()
        return var, standardizer

    def build_model(self):
        with pm.Model() as model:
            # Hyperpriors for group nodes
            eps = pm.HalfCauchy("eps", 4)
            a = pm.Normal('a', mu=-1.3, sd=10)
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = pm.Normal(var_name, mu = -1.1, sd= 5 )
                mu = a + b_i * self.inputs_shared[var_name]

            w = pm.Normal( "w", mu=mu, sigma=eps, observed=self.target)
        self.model = model
        return


    def inference(self):
        self.build_model()
        with self.model:
            step = pm.NUTS(target_accept=0.95)
            idata = pm.sample(1000, step=step, tune=1000, return_inferencedata=True, chains=1)
        self.idata = idata

        idata.to_netcdf(self.save_to/"idata.nc")
    
    def read_inference(self):
        self.idata = az.from_netcdf(self.save_to/"idata.nc")
        return
    
    def counterfactual_plot(self, x, cvar_name, N_samples=100):
        assert len(x) == len(self.target)

        # set control variable
        self.inputs_shared[cvar_name].set_value(x) 

        print("Sample posterior...")
        with self.model:
            post_pred = pm.sample_posterior_predictive(self.idata.posterior)
        print("Done")

        # plot the hdi of the prediction scatter
        _, ax = plt.subplots()
        az.plot_hdi(x, post_pred['w'])
        ax.plot(x, post_pred['w'].mean(0))
        ax.scatter(self.inputs[cvar_name], self.target)
        plt.show()

    def validation(self, config):
        df = az.summary(self.idata)
        # Get the mean value
        a = df.loc['a', 'mean']
        w_preds = a
        for var_name in self.var_names:
            b_i = df.loc[var_name, 'mean']
            w_preds += b_i*self.inputs[var_name]
        
        self.wp['w_pred_mean'] = w_preds


        # Iterate through the input wp
        samples = self.wp['sample'].unique()
        weights = {sample:{} for sample in samples}
        for i in range(len(self.wp)):
            sample = self.wp.loc[i, 'sample']
            mode = self.wp.loc[i, 'mode']
            w_pred = self.wp.loc[i, 'w_pred_mean']
            weights[sample][mode] = w_pred
        
        vs = VisualizeOptimization(config)
        vs.plot_T_probe(theta_deg=0, weights=weights)
        return