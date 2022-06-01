from pathlib import Path
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

from pyStruct.structurePredictor.structures import StructurePredictorInterface
from pyStruct.sampleCollector.sampleStructure import BoundaryCondition, Sample
from pyStruct.sampleCollector.sampleSetStructure import SampleSet
from pyStruct.errors import StructureModelNoneExist, ModelPathNotFound


def get_probability_by_distance(predicted_features:np.ndarray, target_features:np.ndarray):
    norm = StandardScaler()
    target_feature_norm = norm.fit_transform(target_features)
    predicted_features_norm = norm.transform(predicted_features)
    
    distance = np.linalg.norm(predicted_features_norm-target_feature_norm, axis=1)
    prob = 1/distance / sum(1/distance)
    return prob

def find_corresponding_sample(
    predicted_features: np.ndarray, 
    mode: int,
    sample_set: SampleSet,
    ): 
    # get all samples' ref_features
    # ref_features with shape: (N_samples, N_features)
    ref_features = sample_set.flow_features.descriptors[:, mode, :]
    prob = get_probability_by_distance(predicted_features, ref_features)
    sample_id = prob.argmax()
    return sample_set.samples[sample_id]


class GBLookupStructure(StructurePredictorInterface):
    def __init__(self, config, folder: Path):
        self._config = config
        # Initiate model and norm
        self._norm = StandardScaler()
        self._models = []
        self.params = {
            'n_estimators': 100, 
            'max_depth': 7, 
            'min_samples_split': 2, 
            'min_samples_leaf': 1 
            }

        # Model path
        self.save_to = folder / 'model.pkl'

    def train(self, training_samples: SampleSet) -> None:
        """
        For every sample in training samples
        Extract the scalars (descriptors) as targets
        The boundary conditions are the features
        """
        X = training_samples.bc(self._config.x_labels)
        y = training_samples.flow_features.descriptors

        array_features = self._norm.fit_transform(X)
        array_targets = y

        N_modes = self._config.N_modes
        for mode in range(N_modes):
            print(f'train structure predictor mode {mode}')
            x = array_features
            y = array_targets[:, mode, :]
            model = self.create_model()
            model.fit(x, y)
            self._models.append(model)
        return
    
    def create_model(self):
        return MultiOutputRegressor(ensemble.GradientBoostingRegressor(**self.params))

    def predict(self, bc: BoundaryCondition, training_set: SampleSet) -> tuple[np.ndarray, list[Sample]]:
        x = bc.array(self._config.x_labels)
        if len(self._models)  == 0:
            raise StructureModelNoneExist("no models exist. Train structure first.")
        else:
            x = self._norm.transform(x)
            predicted_descriptors = np.array([model.predict(x) for model in self._models])
            predicted_samples = [find_corresponding_sample(
                    predicted_descriptors[mode],
                    mode,
                    training_set,
            ) for mode in range(len(self._models))]
        return  predicted_descriptors, predicted_samples

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

