from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from rainflow import extract_cycles

from pyStruct.machines.feature_processors import *
from pyStruct.machines.optimizers import *
from pyStruct.machines.regressors import *
from pyStruct.machines.structures import *
from pyStruct.machines.reconstructors import *
from pyStruct.data.datastructures import PodSampleSet
from pyStruct.machines.directory import FrameworkPaths
from pyStruct.machines.errors import SampleNotFoundInOptimizationRecord
from pyStruct.config import check_config


FEATURE_PROCESSORS = {
    'COHERENT_STRENGTH': PodCoherentStrength
}

OPTIMIZERS = {
    'POSITIVE_WEIGHTS': PositiveWeights,
    'ALL_WEIGHTS':AllWeights,
    'INTERCEPT': InterceptAndWeights,
    'POS_INTERCEPT': PositiveInterceptAndWeights,
}

WEIGHTS_PREDICTORS ={
    'BAYESIAN': BayesianModel,
    'MULTIBAYESIAN':MultiLevelBayesian,
    'GB': GbRegressor,
    'BSPLINE':BSpline,
    'NN': NnRegressor
}
STRUCTURE_PREDICTORS = {
    'GB': GBLookupStructure,
    'BAYESIAN': BayesianLookupStructure
}
RECONSTRUCTORS = {
    'LINEAR': LinearReconstruction,
    'INTERCEPT': InterceptReconstruction
}



def initialize_pod_samples(workspace, theta_degs, normalize_y):
    """ A messy code... should refactor later"""
    samples = []

    pod_manager = PodModesManager(name='', workspace=workspace, normalize_y=normalize_y)
    bcs = pod_manager._read_signac()
    for sample in range(pod_manager.N_samples):
        m_c = bcs[sample]['M_COLD_KGM3S']
        m_h = bcs[sample]['M_HOT_KGM3S']
        T_c = bcs[sample]['T_COLD_C']
        T_h = bcs[sample]['T_HOT_C']

        X_spatial = pod_manager.X_spatials[sample, ...]
        X_temporal = pod_manager.X_temporals[sample, ...]
        X_s = pod_manager.X_s[sample, ...]
        coord = pod_manager.coords[sample]
        # pod = PodFeatures(coord=coord, X_spatial=X_spatial, X_temporal=X_temporal, X_s=X_s)

        for theta_deg in theta_degs:
            T_wall = pod_manager.T_walls[f'{theta_deg:.2f}']['T_wall'][sample, :]
            loc_index =pod_manager.T_walls[f'{theta_deg:.2f}']['loc_index']
            bc = BoundaryCondition(m_c, m_h, T_c, T_h, theta_deg)
            s = PodSample(bc, loc_index, T_wall)
            s.set_pod(X_spatial, X_temporal, X_s, coord)
            samples.append(s)
    return samples


class TwoMachineFramework:
    def __init__(self, config, start_new=False):
        # Check config
        print(config)
        check_config(config)

        # initialize the machines
        self.config = config

        if start_new:
            shutil.rmtree(Path(self.config.paths.save_to))

        # Create Path manager
        self.paths = FrameworkPaths(root=config.paths.save_to, machine_config=config.machines)

        # initialize samples
        self.samples = initialize_pod_samples(
            config.features.workspace, 
            config.features.theta_degs,
            config.features.normalize_y
            )
        
        np.random.seed(1)
        np.random.shuffle(self.samples)
        N_samples = len(self.samples)
        N_trains = int(N_samples * self.config.features.N_trains_percent)
        print('==== Samples ====')
        print(f'All: {N_samples}')
        print(f'Training: {N_trains}')
        print(f'Testing: {N_samples - N_trains}')

        self.training_set = PodSampleSet(self.samples[:N_trains])
        self.testing_set = PodSampleSet(self.samples[N_trains:])


        # Feature processor 
        self.feature_processor = FEATURE_PROCESSORS[
            config.machines.feature_processor.upper()](
                config.features)
        self.feature_processor.process_samples(self.training_set)

        # Structure predictor
        self.structure_predictor = STRUCTURE_PREDICTORS[
            config.machines.structure_predictor.upper()](
                config.features, self.paths.structure_predictor)

        # Optimization
        self.optimizer = OPTIMIZERS[
            config.machines.optimizer.upper()](
                self.paths.optimizer)

        # Predict weights
        self.weights_predictor = WEIGHTS_PREDICTORS[
            config.machines.weights_predictor.upper()](
                config.features, self.paths.weights_predictor)

        # Reconstructor
        self.reconstructor = RECONSTRUCTORS[
            config.machines.reconstructor.upper()]()

    def _train_structure(self):
        """ """
        # Specify save folder
        if self.structure_predictor.save_to.exists():
            print("Structure model exist; LOAD")
            self.structure_predictor.load()
        else:
            # Get x y pairs and train
            print("Structure model doesn't exist; TRAIN")
            self.structure_predictor.train(self.training_set)
            self.structure_predictor.save()
        return 
    
    def _optimize(self) -> None:
        # Decoupled method: this part is independent of the structure prediction
        # compose temporal matrix
        # by training samples

        N_modes = self.config.features.N_modes
        record_file = self.optimizer.save_to/'optm.csv'
        if record_file.exists():
            record = pd.read_csv(record_file)
            for i, sample in enumerate(self.training_set.samples):
                weights = record.loc[(record['sample'] == sample.name), [f'w{mode}' for mode in range(N_modes)]].values.flatten()
                if len(weights) == 0:
                    raise SampleNotFoundInOptimizationRecord(
                        f"sample name: {sample.name}"
                    )
                sample.set_optimized_weights(weights)
        else:
            all_weights=[]
            for i, sample in enumerate(self.training_set.samples):
                X = sample.flow_features.time_series
                y = sample.wall_true

                # Optimize
                weights = self.optimizer.optimize(X, y) # shape: (N_modes, )
                all_weights.append(weights) 
                sample.set_optimized_weights(weights)
        
            record = pd.DataFrame(np.array(all_weights), columns=[f'w{mode}' for mode in range(len(weights))])
            record.insert(0, 'sample',self.training_set.name)
            record.to_csv(self.optimizer.save_to/'optm.csv', index=False)
        return

    def _train_weights(self):
        # Specify save folder
        if self.weights_predictor.save_to.exists():
            print("Weights model exist; LOAD")
            self.weights_predictor.load()
        else:
            # Get x y pairs and train
            print("Weights model doesn't exist; TRAIN")
            self.weights_predictor.train(self.training_set)
            self.weights_predictor.save()
        
    def train(self):
        # Train Structure
        self._train_structure()
        for sample in self.training_set.samples:
            _, predicted_samples = self.structure_predictor.predict(sample.bc, self.training_set)
            sample.set_pred_structures([s.name for s in predicted_samples])
        assert all(hasattr(sample, 'pred_structures_names') for sample in self.training_set.samples)

        # Train Weights
        # optimize 
        self._optimize()
        self._train_weights()
        for sample in self.training_set.samples:
            N_modes_descriptors = sample.flow_features.descriptors
            weights = self.weights_predictor.predict(N_modes_descriptors)
            sample.set_predicted_weights(weights)
        
        # Full Framework
        for sample in self.training_set.samples:
            y_pred = self.predict(sample.bc)
            sample.set_wall_pred(y_pred)
        return

    def predict(self, bc: BoundaryCondition) -> np.ndarray:
        # Predict structures
        predicted_descriptors, predicted_samples = self.structure_predictor.predict(bc, self.training_set)

        # Predict weights
        weights = self.weights_predictor.predict(predicted_descriptors)
        print(weights.shape)

        # compose
        pred_set = PodSampleSet(predicted_samples)
        X = pred_set.flow_features.time_series

        # reconstruct
        y_pred = self.reconstructor.reconstruct(X, weights)
        return y_pred



class ValidateFramework:
    def __init__(self, sample_set: PodSampleSet, reconstructor):
        self.sample_set = sample_set
        self.reconstructor = reconstructor

    def validate_structure(self):
        for sample in self.sample_set.samples:
            print(f'BC: {sample.name}')
            print(sample.pred_structures_names)
            print("================")
        return 
    
    def validate_optimize(self, samples=None) -> list[PodSample]:
        if samples is None:
            samples = self.sample_set.samples

        for sample in samples:
            weights = sample.w_optm
            # Reconstruct
            X = sample.flow_features.time_series
            y_optm = self.reconstructor.reconstruct(X, weights)
            y_pred = sample.wall_true
            plt.plot(y_pred, label='true')
            plt.plot(y_optm, label='optm')
            plt.show()
        return 

    def validate_weights(self) -> None:
        for sample in self.sample_set.samples:
            plt.scatter(sample.w_optm, sample.w_pred, label=sample.name)
        plt.plot([-1, 1], [-1,1], 'k--')
        plt.legend(bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.show()
        return

    def validate_workflow(self, samples=None):
        if samples is None:
            samples = self.sample_set.samples
        for sample in samples:
            plt.plot(sample.wall_true, label='true')
            plt.plot(sample.wall_pred, label='pred')
            plt.show
        pass