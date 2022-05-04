from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from rainflow import extract_cycles

from pyStruct.database.signacJobData import SignacJobData
from pyStruct.machineSelector import machine_selection
from pyStruct.sampleCollector.sampleStructure import Sample, BoundaryCondition
from pyStruct.sampleCollector.sampleSetStructure import SampleSet
from pyStruct.sampleCollector.samples import initialize_sample_from_signac
from pyStruct.directory import FrameworkPaths
from pyStruct.errors import SampleNotFoundInOptimizationRecord
from pyStruct.config import check_config
from pyStruct.optimizers.optimizer import optimize

def get_sample_flow_features(feature_processor, sample_set):
    for sample in sample_set.samples:
        time_series, descriptors = feature_processor.process_features(sample)
        sample.set_flowfeatures(time_series, descriptors)
    return

def get_sample_pred_structures(structure_predictor, sample_set):
    for sample in sample_set.samples:
        _, predicted_samples = structure_predictor.predict(sample.bc, sample_set)
        sample.set_pred_structures([s.name for s in predicted_samples])
    assert all(hasattr(sample, 'pred_structures_names') for sample in sample_set.samples)
    return

def get_sample_optimization_weights(optimizer, sample_set):
    for i, sample in enumerate(sample_set.samples):
        weights = optimizer.all_weights[i, :]
        sample.set_optimized_weights(weights)
    return

def get_sample_prediction_weights(weights_predictor, sample_set):
    for sample in sample_set.samples:
        N_modes_descriptors = sample.flow_features.descriptors
        weights = weights_predictor.predict(N_modes_descriptors, sample.bc.array())
        sample.set_predicted_weights(weights)
    return

def get_sample_full_prediction(machine, sample_set):
    for sample in sample_set.samples:
        y_pred = machine.predict(sample.bc)
        sample.set_wall_pred(y_pred)
    return

    

def train_structure(structure_predictor, sample_set: SampleSet):
    # Specify save folder
    if structure_predictor.save_to.exists():
        print("Structure model exist; LOAD")
        structure_predictor.load()
    else:
        # Get x y pairs and train
        print("Structure model doesn't exist; TRAIN")
        structure_predictor.train(sample_set)
        structure_predictor.save()
    return

def train_weights(weights_predictor, sample_set: SampleSet):
    # Specify save folder
    if weights_predictor.save_to.exists():
        print("Weights model exist; LOAD")
        weights_predictor.load()
    else:
        # Get x y pairs and train
        print("Weights model doesn't exist; TRAIN")
        weights_predictor.train(sample_set)
        weights_predictor.save()
    return





class TwoMachineFramework:
    def __init__(self, config, start_new=False):
        # Check config
        check_config(config)

        # initialize the machines
        self.config = config

        if start_new:
            shutil.rmtree(Path(self.config.paths.save_to))

        # Create Path manager
        self.paths = FrameworkPaths(root=config.paths.save_to, machine_config=config.machines)

        # initialize machines
        self.initialize_machines(self.config)

        # initialize samples
        self.training_set, self.testing_set = self.initialize_samples(self.config.signac_sample)
        # get_sample_flow_features(self.feature_processor, self.training_set)

    def initialize_samples(self, config):
        samples = initialize_sample_from_signac(config)
        assert len(samples) > 0, "Empty samples, check given condition"
        # Split 
        np.random.seed(1)
        np.random.shuffle(samples)
        N_samples = len(samples)
        N_trains = int(N_samples * self.config.features.N_trains_percent)
        print('==== Samples ====')
        print(f'All: {N_samples}')
        print(f'Training: {N_trains}')
        print(f'Testing: {N_samples - N_trains}')

        training_set = SampleSet(samples[:N_trains])
        testing_set = SampleSet(samples[N_trains:])
        return training_set, testing_set

    def initialize_machines(self, config):
        feature_processor, optimizer, weights_predictor, structure_predictor, reconstructor = machine_selection(config.machines)

        # Feature processor 
        self.feature_processor = feature_processor(
            config.features, 
            self.paths.feature_processor
            )

        # Structure predictor
        self.structure_predictor = structure_predictor(
            config.features, 
            self.paths.structure_predictor
            )

        # Optimization
        self.optimizer = optimizer(
            self.paths.optimizer
            )

        # Predict weights
        self.weights_predictor = weights_predictor(
            config.features, 
            self.paths.weights_predictor
            )

        # Reconstructor
        self.reconstructor = reconstructor()
        return 

    def train(self):
        # Train Structure
        train_structure(self.structure_predictor, self.training_set)
        get_sample_pred_structures(self.structure_predictor, self.training_set)

        # optimize 
        optimize(self.optimizer, self.training_set)
        get_sample_optimization_weights(self.optimizer, self.training_set)

        # Train Weights
        train_weights(self.weights_predictor, self.training_set)
        get_sample_prediction_weights(self.weights_predictor, self.training_set)

        # Keep track
        get_sample_full_prediction(self, self.training_set)
        return

    def test(self):
        get_sample_flow_features(self.feature_processor, self.testing_set)
        get_sample_pred_structures(self.structure_predictor, self.testing_set)
        get_sample_prediction_weights(self.weights_predictor, self.testing_set)
        get_sample_full_prediction(self, self.testing_set)
        return

    def predict(self, bc: BoundaryCondition) -> np.ndarray:
        # Predict structures
        predicted_descriptors, predicted_samples = self.structure_predictor.predict(bc, self.training_set)

        # Predict weights
        weights = self.weights_predictor.predict(predicted_descriptors, bc.array())
        print(weights.shape)

        # compose
        pred_set = SampleSet(predicted_samples)
        # X = pred_set.flow_features.time_series
        X = np.array([s.flow_features.time_series[mode, :] for mode, s in enumerate(predicted_samples) ])

        # reconstruct
        y_pred = self.reconstructor.reconstruct(X, weights)
        return y_pred



class ValidateFramework:
    def __init__(self, sample_set: SampleSet, reconstructor):
        self.sample_set = sample_set
        self.reconstructor = reconstructor

    def validate_structure(self, file=None):
        if file:
            with open(file, 'w') as f:
                for sample in self.sample_set.samples:
                    f.write(f'BC: {sample.name}\n')
                    f.write(','.join(sample.pred_structures_names))
                    f.write("\n================\n")

        else:
            for sample in self.sample_set.samples:
                print(f'BC: {sample.name}')
                print(sample.pred_structures_names)
                print("================")

        return 
    
    def validate_optimize(self, samples: list[Sample]=None, folder=None) -> list[Sample]:
        if samples is None:
            samples = self.sample_set.samples

        for sample in samples:
            weights = sample.w_optm
            # Reconstruct
            X = sample.flow_features.time_series
            y_true = sample.wall_true
            y_optm = self.reconstructor.reconstruct(X, weights)

            plt.plot(y_true, label='true')
            plt.plot(y_optm, label='optm')
            plt.legend()
            plt.title(str(sample.bc))
            

            if folder:
                plt.savefig(folder/f'val_optm_{sample.name}.png')
                plt.clf()
            else:
                plt.show()
        return 

    def validate_weights(self, file=None) -> None:
        for sample in self.sample_set.samples:
            plt.scatter(sample.w_optm, sample.w_pred, label=sample.name)
        plt.plot([-1, 1], [-1,1], 'k--')
        plt.legend(bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        if file:
            plt.savefig(file)
        plt.show()
        return

    def validate_workflow(self, samples=None, folder=None):
        if samples is None:
            samples = self.sample_set.samples
        for sample in samples:
            plt.plot(sample.wall_true, label='true')
            plt.plot(sample.wall_pred, label='pred')
            plt.legend()
            if folder:
                plt.savefig(folder/f'val_{sample.name}.png')
                plt.clf()
            else:
                plt.show()
        pass