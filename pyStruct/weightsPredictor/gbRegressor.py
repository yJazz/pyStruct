import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from scipy.stats import halfcauchy, norm

from pyStruct.weightsPredictor.regressors import WeightsPredictor
from pyStruct.sampleCollector.sampleSetStructure import SampleSet
from pyStruct.errors import ModelPathNotFound


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

    def train(self, sample_set: SampleSet) -> None:
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
