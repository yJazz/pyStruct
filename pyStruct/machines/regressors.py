import pickle
import numpy as np
from typing import Protocol

from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from scipy.stats import halfcauchy, norm
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pyStruct.data.datastructures import PodSampleSet, PodSample, BoundaryCondition
from pyStruct.machines.errors import ModelPathNotFound


class WeightsPredictor:
    def train(self, sample_set):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
        

class GbRegressor(WeightsPredictor):
    def __init__(self, config, folder: Path):
        self.config = config
        self.params = {
            "n_estimators": 1000,
            "max_depth":10,
            "min_samples_split": 2,
            "learning_rate": 0.005,
            "loss": "squared_error",
        }
        self._models = []
        self._norms = [] # since the features are in 3D array, stratify by mode
        self.save_to = folder / 'models.pkl'

    def train(self, sample_set: PodSampleSet) -> None:
        # Get x and y
        features = sample_set.flow_features.descriptors  
        targets = sample_set.w_optm

        for mode in range(self.config.N_modes):
            x = features[:, mode, :]
            norm = MinMaxScaler()
            x = norm.fit_transform(x)

            y = targets[:, mode]
            model = self.create_model()
            model.fit(x, y)
            self._models.append(model)
            self._norms.append(norm)
        return

    def predict(self, N_modes_descriptors: np.ndarray) -> np.ndarray:
        """ 
        input: 
            descriptors: np.ndarray, shape (20, N_features)
        output:
            weight: float
        """
        weights = np.zeros(len(N_modes_descriptors))
        for mode in range(len(N_modes_descriptors)):
            model = self._models[mode]
            norm = self._norms[mode]
            x = norm.transform(N_modes_descriptors[mode, :].reshape(1, -1))
            weights[mode] = model.predict(x)
        return weights
        
    def create_model(self):
        return ensemble.GradientBoostingRegressor(**self.params)

    def save(self):
        if hasattr(self, 'save_to'):
            with open(self.save_to, 'wb') as f:
                pickle.dump((self._models, self._norms), f)
        else:
            raise ModelPathNotFound("Need to initate `set_model_path`")
    
    def load(self):
        if hasattr(self, 'save_to'):
            with open(self.save_to, 'rb') as f:
                self._models, self._norms = pickle.load(f)
        else:
            raise ModelPathNotFound("Need to initate `set_model_path`")




def plot_regression_deviance(reg, params, X_test_norm, y_test):
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test_norm)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show(block=False)

def plot_feature_importance(reg, feature_names, X_test_norm, y_test):

    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        reg, X_test_norm, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show(block=False)

def plot_weights_accuracy_scatter(reg, X_train_norm, y_train, X_test_norm, y_test):

    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes[0]
    y_pred = reg.predict(X_train_norm)
    y_true = y_train
    ax.scatter(y_true, y_pred)
    ax.plot([-3,2], [-3,2], 'k')
    ax.set_title("Train")

    ax = axes[1]
    y_pred = reg.predict(X_test_norm)
    y_true = y_test
    ax.scatter(y_true, y_pred)
    ax.plot([-3,2], [-3,2], 'k')
    ax.set_title("Test")
    plt.show(block=False)


def gb_regression(params, X_train, y_train, X_test, y_test, feature_names, show=False):
    # Normalize the features 
    norm = MinMaxScaler().fit(X_train.to_numpy())
    X_train_norm = norm.transform(X_train.to_numpy())
    X_test_norm = norm.transform(X_test.to_numpy())


    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train_norm, y_train)
    error = mean_squared_error(y_test, reg.predict(X_test_norm))
    print(f"Regression mean squre error: {error}")
    if show:
        plot_regression_deviance(reg, params, X_test_norm, y_test)
        plot_feature_importance(reg, feature_names, X_test_norm, y_test)
        plot_weights_accuracy_scatter(reg, X_train_norm, y_train, X_test_norm, y_test)
    return reg, norm



def split_mode_traintest(wp_mode, train_ratio):
    N_s = len(wp_mode)
    wp_train = wp_mode.iloc[:int(train_ratio*N_s)]
    wp_test = wp_mode.iloc[int(train_ratio*N_s):]
    return wp_train, wp_test


def get_regressors_for_each_mode(N_modes, params, wp, feature_names, train_ratio=1, show=False):
    regs = []
    norms = []
    for mode in range(N_modes):
        print(f"---regression mode:{mode}---")
        wp_mode = wp[wp['mode'] == mode]
        wp_train, wp_test = split_mode_traintest(wp_mode.sample(frac=1), train_ratio)
        print(wp_train)

        # Prepare train and test
        X_train = wp_train[feature_names]
        y_train = wp_train['w']
        X_test = wp_test[feature_names]
        y_test = wp_test['w']

        reg, norm = gb_regression(params, X_train, y_train, X_test, y_test, feature_names, show)
        regs.append(reg)
        norms.append(norm)
    
    if show:
        plt.show(block=True)
    return regs, norms

def wp_split_train_test(feature_table, train_id, feature_labels, target_label):
    bool_series = feature_table['sample'].isin(train_id)
    table_train = feature_table[bool_series]
    table_test = feature_table[bool_series==False]
    X_train = table_train[feature_labels]
    y_train = table_train[target_label]
    X_test = table_test[feature_labels]
    y_test = table_test[target_label]
    return X_train, y_train, X_test, y_test




class GbRegressor_repr(WeightsPredictor):
    def __init__(self, config):
        self.config = config
        self.params = {
            "n_estimators": 1000,
            "max_depth":10,
            "min_samples_split": 2,
            "learning_rate": 0.005,
            "loss": "squared_error",
        }
        self.save_folder = Path(self.config['save_to']) / 'weights_predictor'
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder / "gb_regressor.pkl"
    
    def train(self, feature_table, show=False):
        self.regs = []
        self.norms = []
        for rank in range(self.config['N_modes']):
            f = feature_table[feature_table['rank'] == rank]
            X_train, y_train, X_test, y_test = wp_split_train_test(
                f,
                self.config['training_id'],
                self.config['y_labels'],
                ['w'])


            norm = MinMaxScaler().fit(X_train.to_numpy())
            X_train_norm = norm.transform(X_train.to_numpy())
            X_test_norm = norm.transform(X_test.to_numpy())


            reg = ensemble.GradientBoostingRegressor(**self.params)
            reg.fit(X_train_norm, y_train)
            error = mean_squared_error(y_test, reg.predict(X_test_norm))
            print(f"Regression mean squre error: {error}")

            self.regs.append(reg)
            self.norms.append(norm)

            if show:
                # plot_regression_deviance(reg, self.params, X_test_norm, y_test)
                # plot_feature_importance(reg, feature_names, X_test_norm, y_test)
                plot_weights_accuracy_scatter(reg, X_train_norm, y_train, X_test_norm, y_test)

        # Save 
        with open(self.save_as, 'wb') as f:
            pickle.dump((self.regs, self.norms), f)
        return feature_table
    
    def load(self, feature_table):
        print("load file")
        with open(self.save_as, 'rb') as f:
            self.regs, self.norms = pickle.load(f)
        return


    def predict(self, features: np.ndarray):
        assert hasattr(self, 'regs'), "No regressors exist. Train or Read first"
        assert features.shape == (self.config['N_modes'], len(self.config['y_labels']))
            
        weights=np.zeros(self.config['N_modes'], dtype=np.float)
        for mode in range(self.config['N_modes']):
            reg = self.regs[mode]
            norm = self.norms[mode]
            # normalize the feature
            normalized_feature_mode = norm.transform(features)[mode, :]
            weights[mode] = reg.predict(normalized_feature_mode.reshape(1,len(self.config['y_labels'])))
            # weights.append(reg.predict(normalized_feature_mode.reshape(1,len(self.config['y_labels']))))
        return weights



class NnRegressor(WeightsPredictor):
    """Give feature"""
    def __init__(self, config: dict):

        import keras
        from keras.models import Model
        from keras.layers import Input, Dense, Dropout
        from sklearn.preprocessing import StandardScaler
        self.config = config

        self.save_folder = Path(self.config['save_to']) / 'NN_weights_predictor'
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder / "NN_regressor"

    def _create_model(self):
        dropout_rate = 0.25
        inputs = Input(shape=(len(self.config['y_labels']),))
        x = Dense(64, activation='relu')(inputs)
        # x = Dropout(dropout_rate)(x, training=True)
        x = Dense(128, activation='relu')(x)
        # x = Dropout(dropout_rate)(x, training=True)
        x = Dense(256, activation='relu')(x)
        # x = Dropout(dropout_rate)(x, training=True)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs, outputs)    

        return model

    def _pre_processing(self, feature_table: pd.DataFrame):
        dataset = feature_table.copy()
        # Normalize 
        self.normalizer = StandardScaler()
        norm_features = self.normalizer.fit_transform(dataset[self.config['y_labels']])
        for i, label in enumerate(self.config['y_labels']):
            dataset[f'{label}'] = norm_features[:, i]

        # Split into train, test 
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # Specify features and labels
        train_features = train_dataset[self.config['y_labels']]
        test_feature = test_dataset[self.config['y_labels']]

        train_labels = train_dataset['w']
        test_labels = test_dataset['w']
        return train_features, train_labels, test_feature, test_labels
    
    def train(self, feature_table: pd.DataFrame, create_model=True, epochs=1000):
        train_features, train_labels, test_feature, test_labels = self._pre_processing(feature_table)
        if create_model:
            self.model = self._create_model()
            opt = keras.optimizers.Adam(learning_rate=0.005)
            self.model.compile(loss='mean_squared_error', optimizer=opt)
        self.model.fit(train_features, train_labels, epochs=epochs, validation_split=0.2)

        # Save 
        self.model.save(self.save_as)
        with open(self.save_as/'norm.pkl', 'wb') as f:
            pickle.dump(self.normalizer, f)
        return

    def load(self, feature_table):
        print("load file")
        self.model = keras.models.load_model(self.save_as)
        with open(self.save_as/'norm.pkl', 'rb') as f:
            self.normalizer = pickle.load(f)

    def predict(self, features: np.ndarray, samples=100):
        # Normalize features
        features = self.normalizer.transform(features)
        # predictions = []
        # for i in range(samples):
            # predictions.append(self.model.predict(features))
        predictions = self.model.predict(features)
        return predictions.flatten()
        


    
class BayesianModel(WeightsPredictor):
    def __init__(
        self, 
        config:dict,
        ):

        import arviz as az 
        import pymc3 as pm
        from patsy import dmatrix, build_design_matrices
        import scipy.stats as stats
        from theano import shared
        from sklearn.preprocessing import StandardScaler
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


