from pathlib import Path
import pickle
import pandas as pd
import numpy as np

import arviz as az 
import pymc3 as pm
from patsy import dmatrix, build_design_matrices
import scipy.stats as stats
from theano import shared
from sklearn.preprocessing import StandardScaler

from pyStruct.weightsPredictor.regressors import WeightsPredictor

class BayesianModel(WeightsPredictor):
    def __init__(
        self, 
        config:dict,
        ):

        self.config = config
        self.var_names = config['y_labels']
        self.target_name = 'w'

        self.save_folder = Path(config['save_to'])/f"bayesian_model"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder/"idata.nc"

    def _init_data(self, feature_table):
        # get data
        self.feature_table = feature_table
        self.target = self.feature_table[self.target_name].values
        self.inputs = {}
        self.inputs_shared={}
        self.standrdizers = {}
        for var_name in self.var_names:
            input, standrdizer = self.transform_input(var_name)
            self.inputs_shared[var_name] = shared(input) # the shared tensor, which can be modified later
            self.standrdizers[var_name] = standrdizer
            self.inputs[var_name] = input # this won't change

    def build_model(self):
        with pm.Model() as model:
            # Hyperpriors for group nodes
            eps = pm.HalfCauchy("eps", 4)
            a = pm.Normal('a', mu=0.5, sd=1)
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = pm.Normal(var_name, mu = -1.1, sd= 5 )
                mu += b_i * self.inputs_shared[var_name]
            w = pm.Normal( "w", mu=mu, sigma=eps, observed=self.target)
        self.model = model
        return

    def transform_input(self, var_name):
        standardizer = StandardScaler()
        var = standardizer.fit_transform(self.feature_table[var_name].values.reshape(-1,1)).flatten()
        return var, standardizer
    
    def train(self, feature_table: pd.DataFrame, show=False, samples=1000):
        """ Inference """
        self._init_data(feature_table)
        self.build_model()
        with self.model:
            # step = pm.NUTS(target_accept=0.95)
            idata = pm.sample(samples, tune=1000, return_inferencedata=True, chains=1)
        self.idata = idata
        self.df = az.summary(idata)
        idata.to_netcdf(self.save_as)
        with open(self.save_folder / "standardizer.pkl", 'wb') as f:
            pickle.dump(self.standrdizers, f)
        return 

    def load(self, feature_table):
        print("load Bayesian weights predictor")
        self._init_data(feature_table)
        self.build_model()
        self.idata = az.from_netcdf(self.save_folder/"idata.nc")
        self.df = az.summary(self.idata)
        with open(self.save_folder / "standardizer.pkl", 'rb') as f:
            self.standrdizers = pickle.load(f)
        return

    def predict(self, features):
        assert features.shape == (self.config['N_modes'], len(self.config['y_labels']))
        # features = features.to_numpy()

        # Get the mean value
        a = self.df.loc['a', 'mean']
        weights =[]
        for mode in range(self.config['N_modes']):
            w_preds = a
            for i, var_name in enumerate(self.var_names):
                b_i = self.df.loc[var_name, 'mean']
                w_preds += b_i*self.standrdizers[var_name].transform(features[mode, i].reshape(-1, 1))
            
            weights.append(w_preds[0])
        return weights
    def sampling(self, features, N_realizations=100):

        # Get the mean value
        sigma = self.df.loc['eps', 'mean']
        a = self.df.loc['a', 'mean']
        weights_samples = np.zeros((self.config['N_modes'], N_realizations ))
        for mode in range(self.config['N_modes']):
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = self.df.loc[var_name, 'mean']
                mu += b_i*self.standrdizers[var_name].transform(features[mode, i].reshape(-1, 1))

            w_preds = norm.rvs(loc=mu, scale=sigma, size=(1,N_realizations))
            # weights_samples.append(w_preds)
            weights_samples[mode, :] = w_preds.flatten()
        return weights_samples

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
            # w_preds += b_i*self.standrdizers[var_name].transform(self.inputs[var_name])
            w_preds += b_i*self.inputs[var_name]
        
        self.feature_table['w_pred_mean'] = w_preds


        # Iterate through the input wp
        samples = self.feature_table['sample'].unique()
        weights = {sample:{} for sample in samples}
        for i in range(len(self.feature_table)):
            sample = self.feature_table.loc[i, 'sample']
            mode = self.feature_table.loc[i, 'mode']
            w_pred = self.feature_table.loc[i, 'w_pred_mean']
            weights[sample][mode] = w_pred

        print(self.feature_table)
        print(weights[sample])
        
        return

class MultiLevelBayesian(BayesianModel):
    def __init__(self, config: dict):
        super(MultiLevelBayesian, self).__init__(config)
        self.save_folder = Path(config['save_to'])/f"multibayesian_model"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder/"idata.nc"

    def _init_data(self, feature_table):
        # get data
        self.feature_table = feature_table
        self.modes = feature_table['mode'].values
        self.target = self.feature_table[self.target_name].values
        self.inputs = {}
        self.inputs_shared={}
        self.standrdizers = {}
        for var_name in self.var_names:
            input, standrdizer = self.transform_input(var_name)
            self.inputs_shared[var_name] = shared(input) # the shared tensor, which can be modified later
            self.standrdizers[var_name] = standrdizer
            self.inputs[var_name] = input # this won't change
        
    
    def build_model(self):
        """ Multi-level model; split by the mode """
        with pm.Model() as model:
            # Hyperpriors for group nodes
            mu_a = pm.Normal("mu_a", mu=1, sigma=5)
            sigma_a = pm.HalfNormal("sigma_a", 5)
            mu_b = pm.Normal("mu_b", mu=-1, sigma=5)
            sigma_b = pm.HalfNormal("sigma_b", 5)

            # Transformed:
            a_offset = pm.Normal('a_offset', mu=0, sd=1, shape=20)
            a = pm.Deterministic("a", mu_a + a_offset * sigma_a)
            b_offset = pm.Normal('b_offset', mu=0, sd=1, shape=20)
            b = pm.Deterministic("b", mu_b + b_offset * sigma_b)
            
            eps = pm.HalfCauchy("eps", 4)
            mu = a[self.modes] + b[self.modes] * self.inputs_shared['singular']
            w = pm.Normal( "w", mu=mu, sigma=eps, observed=self.target)
        self.model = model
        return

    def predict(self, features: np.ndarray):
        assert features.shape == (self.config['N_modes'], len(self.config['y_labels']))

        # Get the mean value
        weights =[]
        for mode in range(self.config['N_modes']):
            a = self.df.loc[f'a[{mode}]', 'mean']
            b = self.df.loc[f'b[{mode}]', 'mean']
            w_preds = a + b * self.standrdizers['singular'].transform(features[mode].reshape(-1, 1))
            
            weights.append(w_preds[0])
        return weights

    def sampling(self, features, N_realizations=100):
        # Get the mean value
        sigma = self.df.loc['eps', 'mean']
        weights_samples = np.zeros((self.config['N_modes'], N_realizations ))
        for mode in range(self.config['N_modes']):
            a = self.df.loc[f'a[{mode}]', 'mean']
            b = self.df.loc[f'b[{mode}]', 'mean']
            mu = a + b * self.standrdizers['singular'].transform(features[mode].reshape(-1, 1))

            w_preds = norm.rvs(loc=mu, scale=sigma, size=(1,N_realizations))
            # weights_samples.append(w_preds)
            weights_samples[mode, :] = w_preds.flatten()
        return weights_samples

    
class BSpline(WeightsPredictor):
    def __init__(self, config: dict):
        # super(MultiLevelBayesian, self).__init__(config)
        self.config = config
        self.save_folder = Path(config['save_to'])/f"BSpline"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder/"idata_mode_1.nc"

    def build_model(self):
        self.models = [] 
        self.norms =[]
        self.idatas=[]
        self.dfs = []
        

        for rank in range(self.config['N_modes']):
            sub_ft  = self.feature_table[self.feature_table['rank'] == rank]
            norm = StandardScaler()
            singular = norm.fit_transform(sub_ft['singular'].values.reshape(-1,1)).flatten()
            self.norms.append(norm)

            # Define Bspline
            num_knots = 5
            self.N_knots = num_knots
            knot_list = np.quantile(singular, np.linspace(0, 1, num_knots))

            B = dmatrix(
                "bs(singular, knots=knots, degree=3, include_intercept=True) - 1",
                {"singular": singular, "knots": knot_list[1:-1]},
            )

            with pm.Model() as model:
                a = pm.Normal("a", 1, 10)
                w = pm.Normal("w", mu=0, sd=10, shape=B.shape[1])
                mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
                sigma = pm.Exponential("sigma", 1)
                D = pm.Normal("D", mu, sigma, observed=sub_ft['w'].values)
        
        return

    def train(self, feature_table: pd.DataFrame, samples=1000):
        self.feature_table = feature_table
        self._init_data(feature_table)
        self.build_model()

        for i, model in enumerate(self.models):
            with model:
                step = pm.NUTS(target_accept=0.95)
                idata = pm.sample(samples, step=step, tune=1000, return_inferencedata=True, chains=1)

                self.idatas.append(idata)
                idata.to_netcdf(self.save_folder/f'idata_mode_{i}.nc')
                self.idatas.append(idata)
                self.dfs.append(az.summary(idata))
        return
    
    def load(self, feature_table: pd.DataFrame):
        print("Load BSpline")
        self.feature_table = feature_table
        self.build_model()

        for mode in range(self.config['N_modes']):
            self.idatas.append( az.from_netcdf(self.save_folder/f'idata_mode_{mode}.nc'))
            self.dfs.append( az.summary(self.idatas[mode]))
    
    def predict(self, features):
        assert features.shape == (self.config['N_modes'], 
                                  len(self.config['y_labels']))
        weights = []
        
        for rank in range(self.config['N_modes']):
            df = self.dfs[rank]
            a = df.loc['a', 'mean']
            w = np.array([df.loc[f'w[{i}]', 'mean'] for i in range(self.N_knots+2)])

            singular = features[rank, 0]
            sub_ft = self.feature_table[self.feature_table['rank'] == rank]
            singular_ref = sub_ft['singular'].values
            knot_list = np.quantile(singular_ref, np.linspace(0, 1, self.N_knots))

            B = dmatrix(
                "bs(singular, knots=knots, degree=3, include_intercept=True) - 1",
                {"singular": singular_ref, "knots": knot_list[1:-1]},
            )
            quntile = percentileofscore(singular_ref, singular)
            new_design = {"singular": singular, 'knots': knot_list[1:-1]} 
            # Construct a new design matrix
            
            B_new = build_design_matrices([B.design_info], new_design)[0]

            weights.append( a + sum(np.matmul(B_new, w)))
        return weights
