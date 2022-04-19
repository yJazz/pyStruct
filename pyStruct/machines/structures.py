from dataclasses import dataclass, field
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from typing import Protocol


from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import ensemble


from pyStruct.data.dataset import read_pickle
from pyStruct.machines.errors import LabelNotInStructureTable, StructureModelNoneExist, ModelPathNotFound
from pyStruct.machines.datastructures import BoundaryCondition


class Model(Protocol):
    """ A protocol """
    def fit(self):
        pass

class Normalizer(Protocol):
    """ A normalizer protocol"""
    def fit_transform(self):
        pass

class StructXyPairs:
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets


@dataclass
class StructPred:
    bc: BoundaryCondition
    mode: int

def get_probability_by_distance(predicted_features:np.ndarray, target_features:np.ndarray):
    distance = np.linalg.norm(target_features.to_numpy()-predicted_features, axis=1)
    prob = 1/distance / sum(1/distance)
    return prob

def find_corresponding_mode(
    predicted_features: np.ndarray, 
    mode:int, 
    feature_names: list[str],
    structure_library: pd.DataFrame,
    ) -> StructPred:
    """ Find the matched component"""
    target_features = structure_library[structure_library['mode'] == mode][feature_names]
    idx = target_features.index
    prob = get_probability_by_distance(predicted_features, target_features)
    target_id = idx[prob.argmin()]

    keys = BoundaryCondition.__annotations__.keys()
    bc_values = structure_library.loc[target_id, keys].values[:-1]
    bc = BoundaryCondition(*bc_values)
    mode = structure_library.loc[target_id, 'mode']
    return StructPred(bc=bc, mode=mode)





class StructurePredictorInterface:
    def set_model_path(self, to_folder: Path) ->None:
        raise NotImplementedError()
    
    def _check_labels(self, structure_table: pd.DataFrame)->None:
        """ The labels """
        raise NotImplementedError()

    def create_model(self):
        raise NotImplementedError()

    def train(self, structure_table: pd.DataFrame) -> None:
        raise NotImplementedError()
    
    def save(self, save_to: str) -> None: 
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def predict(self, bc: BoundaryCondition) -> np.ndarray:
        raise NotImplementedError()

class GeneralPredictor(StructurePredictorInterface):
    def __init__(self, structure_config: dict, structure_library: pd.DataFrame):
        self._structure_config = structure_config
        self._structure_library = structure_library
        # Initiate model and norm
        self._norm = StandardScaler()
        self._models = []

    def _check_labels(self, x_labels: dict, y_labels: dict, structure_table: pd.DataFrame) -> None:
        """ Make sure the x_labels and y_labels are in structure table"""
        # Check x
        if not all(item in structure_table.keys() for item in x_labels):
            raise LabelNotInStructureTable(
                message = f" Check the y label. The strubure_table keys are: {structure_table.keys()}"
            )
        # Check y
        if not all(item in structure_table.keys() for item in y_labels):
            raise LabelNotInStructureTable(
                message = f" Check the x label. The strubure_table keys are: {structure_table.keys()}"
            )
        return
        
    def train(self) -> None:
        """  """
        self._check_labels(self._structure_config.x_labels, self._structure_config.y_labels, self._structure_library)
        array_features = self._norm.fit_transform(self._structure_library[self._structure_config.x_labels])
        array_targets = self._structure_library[self._structure_config.y_labels].to_numpy()
        xy_pairs =StructXyPairs(array_features, array_targets)

        # train 
        N_modes = len(self._structure_library['mode'].unique())
        for mode in range(N_modes):
            print(f'train structure predictor mode {mode}')
            bool_filter = self._structure_library['mode'] == mode
            x = array_features[bool_filter, :]
            y = array_targets[bool_filter, :]

            model = self.create_model()
            model.fit(x, y)
            self._models.append(model)
        return

    def save(self) -> None:
        if hasattr(self, 'save_to'):
            with open(self.save_to, 'wb') as f:
                pickle.dump((self._models, self._norm), f)
        else:
            raise ModelPathNotFound("Need to initate `set_model_path`")

    def load(self) -> None:
        if hasattr(self, 'save_to'):
            with open(self.save_to, 'rb') as f:
                self._models, self._norm = pickle.load(f)
        else:
            raise ModelPathNotFound("Need to initate `set_model_path`")



class GBLookupStructure(GeneralPredictor):
    def __init__(self, structure_config, structure_library):
        super(GBLookupStructure, self).__init__(structure_config, structure_library)
        self.params = {
            'n_estimators': 100, 
            'max_depth': 7, 
            'min_samples_split': 2, 
            'min_samples_leaf': 1 
            }

    def set_model_path(self, to_folder: Path):
        to_folder.mkdir(parents=True, exist_ok=True)
        self.save_to = to_folder / 'GB.pkl'
    
    def create_model(self):
        return MultiOutputRegressor(ensemble.GradientBoostingRegressor(**self.params))

    def predict(self, bc: BoundaryCondition) -> list[StructPred]:
        x = bc.array(self._structure_config.x_labels)
        if len(self._models)  == 0:
            raise StructureModelNoneExist("no models exist. Train structure first.")
        else:
            x = self._norm.transform(x)
            predictions = [find_corresponding_mode(
                    model.predict(x).flatten(), 
                    mode,
                    self._structure_config.y_labels,
                    self._structure_library
            ) for mode, model in enumerate(self._models)]
        return  predictions




class GBLookupStructure_depr(GeneralPredictor):
    """ Given x_labels, output y_labels
        stratified by modes: each mode: a regression model
    """

    def __init__(self, config):
        self.config = config
        self.params = {
            'n_estimators': 100, 
            'max_depth': 7, 
            'min_samples_split': 2, 
            'min_samples_leaf': 1 
            }

        self.models = []
        self.norm = StandardScaler()
        self.save_folder = Path(self.config['save_to']) / 'structure_predictor'
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder / "lookup_structures.pkl"

    
    def train(self, feature_table):
        self.feature_table = feature_table
        features = feature_table[self.config['x_labels']]
        features_x = self.norm.fit_transform(features)

        targets = feature_table[self.config['y_labels']].to_numpy()

        for mode in range(self.config['N_modes']):
            print(f'train structure predictor mode {mode}')
            x = features_x[feature_table['mode'] == mode, :]
            y = targets[feature_table['mode'] == mode, :]

            model = self._create_model_and_fit(x, y)
            self.models.append(model)
        
        # Save the models
        with open(self.save_as, 'wb') as f:
            pickle.dump((self.models, self.norm), f)

    def load(self, feature_table):
        self.feature_table = feature_table
        print("load lookup-structure models")
        with open(self.save_as, 'rb') as f:
            self.models, self.norm = pickle.load(f)
        return

    def _create_model_and_fit(self, X, y):
        model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**self.params))
        model.fit(X, y)
        return model

    def predict_and_compose(self, inputs: dict, X_base: np.ndarray, perturb_structure:bool=False):
        """ With the given inputs, give out the possible modes,
            and compose new temporal bases X_compose

        Parameters
        ----------
        inputs:  boundary conditions. The keys should be the same as self.config['x_labels']
        X_base: {array-like, with the shape (N_libarary_samples, N_libarary_modes)}
        perturb_structure: if True, do random sampling of structures based on its probability

        Returns
        -------
        X_compose: {array-like, with shape (self.config['N_modes'], self.config['N_t'])}
        predicted_features: {array-like, with shape (self.config['N_modes'], self.config['y_labels'])}, 
                            the features to be fed into the next machine

        """
        X_compose = np.zeros(X_base.shape[1:])

        if perturb_structure:
            predicted_df = self._unc_predict(inputs)
        else:
            predicted_df = self._predict(inputs)
        predicted_features = predicted_df[self.config['y_labels']].to_numpy()

        # Compose predicted bases
        for i in predicted_df.index:
            sample = predicted_df.loc[i, 'sample']
            ranked_mode = int(predicted_df.loc[i, 'rank'])
            X_compose[ranked_mode, :] = X_base[sample, ranked_mode, :]
        return X_compose, predicted_features

    def _predict(self, inputs: dict) -> pd.DataFrame:
        """ Given the inputs, deterministically predict the modes' feature table

        Parameters
        ----------
        inputs: {dict}, the keys should be the same as self.config['x_labels']

        Returns
        -------
        predicted_modes_features_as_table: {pd.DataFrame} of lenth: self.config['N_modes]
        """
        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        
        idx = []
        for mode in range(self.config['N_modes']):
            model = self.models[mode]
            y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]
            y_label_pred = model.predict(x)
            d = self._compare_distance(y_label_pred, y_label)
            id = y_label.index[d.argmin()]
            idx.append(id)
        ranked_modes_features_as_table = self.feature_table.iloc[idx].sort_values(by=['rank'])
        return ranked_modes_features_as_table

    def _unc_predict(self, inputs: dict):
        """ Given the inputs, PROBABILISTICALLY predict the modes' feature table

        Parameters
        ----------
        inputs: {dict}, the keys should be the same as self.config['x_labels']

        Returns
        -------
        predicted_modes_features_as_table: {pd.DataFrame} of lenth: self.config['N_modes]
        """ 
        modified_table = self.feature_table.copy()

        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        idx = []
        for mode in range(self.config['N_modes']):
            # y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]

            y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]

            # Get prediction
            model = self.models[mode]
            y_label_pred = model.predict(x)
            
            # modify table
            d = self._compare_distance(y_label_pred, y_label)
            prob = 1/d / sum(1/d)
            id_candidates = y_label.index

            # Sample those probability
            random_choice = np.random.choice(a=id_candidates, p=prob)
            idx.append(random_choice)
            modified_table.loc[random_choice, self.config['y_labels']] = y_label_pred.flatten()
        
        ranked_modes_features_as_table = modified_table.iloc[idx].sort_values(by=['rank'])
            
        return ranked_modes_features_as_table

    def get_library_probability(self, inputs:dict, N_lib_samples: int, N_lib_modes: int, show=True):
        """
        Get the probability 
        Returns
        --------
        probs: {array-like, with shape (N_lib_samples, N_lib_modes)}
        """
        probs = np.zeros((N_lib_samples, N_lib_modes))
        for mode in range(self.config['N_modes']):
            probs[:, mode] = self._compute_prob_of_mode(inputs, mode)
        
        if show:
            fig, ax = plt.subplots(figsize=(8,4)) 
            sns.heatmap(probs, cmap='RdPu', vmin=0, vmax=1)
            plt.xlabel("Tempoeral Mode")
            plt.ylabel("Training ID")
            plt.show()
        return probs
    
    def _compare_distance(self, y_pred: np.ndarray, y_label: np.ndarray):
        """ The y_pred: shape (1, N_ylabels) """
        assert y_pred.shape == (1, len(self.config['y_labels']))
        return np.linalg.norm( y_label - y_pred, axis=1)
            
    def _compare_distance_and_return_id(self, y_pred: np.ndarray, y_label: np.ndarray):
        """ The y_pred: shape (1, N_ylabels) """
        distance = self._compare_distance(y_pred, y_label)
        id = distance.argmin() 
        return y_label.index[id]

    def _compute_prob_of_mode(self, inputs, mode):
        input_df = pd.DataFrame(inputs, index=[0])
        x = self.norm.transform(input_df)
        y_label = self.feature_table[self.feature_table['mode']== mode][self.config['y_labels']]

        # Get prediction
        model = self.models[mode]
        y_label_pred = model.predict(x)
        
        # modify table
        d = self._compare_distance(y_label_pred, y_label)
        prob = 1/d / sum(1/d)
        return prob 
            

class BayesianLookupStructure(GeneralPredictor):
    def __init__(
        self, 
        config:dict,
        ):

        import arviz as az 
        import pymc3 as pm
        import scipy.stats as stats
        from theano import shared
        from sklearn.preprocessing import StandardScaler
        self.config = config
        self.var_names = config['x_labels']
        self.target_name = 'singular'

        self.save_folder = Path(config['save_to'])/f"bayesian_singular"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder/"idata.nc"

    def _init_data(self, feature_table):
        # get data
        self.feature_table = feature_table
        self.target, self.norm_target = self.transform_input(self.target_name)
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
            a = pm.Normal('a', mu=1, sd=1)
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = pm.Normal(var_name, mu = 0, sd= 10 )
                mu = a + b_i * self.inputs_shared[var_name]
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
        idata.to_netcdf(self.save_as)
        with open(self.save_folder / "standardizer.pkl", 'wb') as f:
            pickle.dump(self.standrdizers, f)
        return 

    def load(self, feature_table):
        print("load Bayesian weights predictor")
        self._init_data(feature_table)
        self.build_model()
        self.idata = az.from_netcdf(self.save_folder/"idata.nc")
        with open(self.save_folder / "standardizer.pkl", 'rb') as f:
            self.standrdizers = pickle.load(f)
        return

    def _predict(self, inputs: dict):
        """ Give the features, """

        # Get the model parameters
        df = az.summary(self.idata)
        # the mean value
        a = df.loc['a', 'mean']

        pred = a
        for i, var_name in enumerate(self.var_names):
            b_i = df.loc[var_name, 'mean']
            pred += b_i*self.standrdizers[var_name].transform(np.array(inputs[var_name]).reshape(-1, 1))
        # Rescale 
        pred = self.norm_target.inverse_transform(pred)

        # Finde the modes 
        idx = []
        for mode in range(self.config['N_modes']):
            # todo 
            pass

        return 

    def predict_and_compose(self, inputs: dict, X_base: np.ndarray, perturb_structure: bool):
        X_compose = np.zeros(X_base.shape[1:])
        predicted_features = self._predict(inputs)

        # Compose 
        for i in predicted_df.index:
            sample = predicted_df.loc[i, 'sample']
            mode = predicted_df.loc[i, 'mode']
            X_compose[mode, :] = X_base[sample, mode, :]
        return X_compose, predicted_features

    def sampling(self, features, N_realizations=100):
        df = az.summary(self.idata)

        # Get the mean value
        sigma = df.loc['eps', 'mean']
        a = df.loc['a', 'mean']
        weights_samples =[]
        for mode in range(self.config['N_modes']):
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = df.loc[var_name, 'mean']
                mu += b_i*self.standrdizers[var_name].transform(features[mode, i].reshape(-1, 1))

            w_preds = norm.rvs(loc=mu, scale=sigma, size=(1,N_realizations))
            weights_samples.append(w_preds)
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
