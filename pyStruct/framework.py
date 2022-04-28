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

        # initialize samples
        samples = initialize_sample_from_signac(self.config.signac_sample)
        assert len(samples) > 0, "Empty samples, check given condition"
        
        np.random.seed(1)
        np.random.shuffle(samples)
        N_samples = len(samples)
        N_trains = int(N_samples * self.config.features.N_trains_percent)
        print('==== Samples ====')
        print(f'All: {N_samples}')
        print(f'Training: {N_trains}')
        print(f'Testing: {N_samples - N_trains}')

        self.training_set = SampleSet(samples[:N_trains])
        self.testing_set = SampleSet(samples[N_trains:])


        # Feature processor 

        feature_processor, optimizer, weights_predictor, structure_predictor, reconstructor = machine_selection(config.machines)
        self.feature_processor = feature_processor(config.features, self.paths.feature_processor)
        for sample in self.training_set.samples:
            print(f"processing sample: {sample.name}")
            timeseries, descriptors = self.feature_processor.process_features(sample)
            sample.set_flowfeatures(timeseries, descriptors)

        # Structure predictor
        self.structure_predictor = structure_predictor(config.features, self.paths.structure_predictor)

        # Optimization
        self.optimizer = optimizer(self.paths.optimizer)

        # Predict weights
        self.weights_predictor = weights_predictor(config.features, self.paths.weights_predictor)

        # Reconstructor
        self.reconstructor = reconstructor()

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
    
    def _optimize(self) -> np.ndarray:
        # Decoupled method: this part is independent of the structure prediction
        # compose temporal matrix
        # by training samples

        N_modes = self.config.features.N_modes
        record_file = self.optimizer.save_to/'optm.csv'
        all_weights=[]
        if record_file.exists():
            print(f"Optimization exist, LOAD")
            record = pd.read_csv(record_file)
            for i, sample in enumerate(self.training_set.samples):
                w = record.loc[(record['sample'] == sample.name)].values.flatten()[1:]
                if len(w) == 0:
                    raise SampleNotFoundInOptimizationRecord(
                        f"sample name: {sample.name}"
                    )
                all_weights.append(w) 
            all_weights = np.array(all_weights)

        else:
            for i, sample in enumerate(self.training_set.samples):
                print(f'Optimize sample: {sample.name}')
                X = sample.flow_features.time_series
                y = sample.wall_true

                # Optimize
                w = self.optimizer.optimize(X, y) # shape: (N_modes, )
                all_weights.append(w) 
            all_weights = np.array(all_weights)
        
            record = pd.DataFrame(all_weights, columns=[f'w{mode}' for mode in range(all_weights.shape[1])])
            record.insert(0, 'sample',self.training_set.name)
            record.to_csv(self.optimizer.save_to/'optm.csv', index=False)
        
        return all_weights

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

        # optimize 
        all_weights = self._optimize()
        for i, sample in enumerate(self.training_set.samples):
            weights = all_weights[i, :]
            sample.set_optimized_weights(weights)

        # Train Weights
        self._train_weights()
        for sample in self.training_set.samples:
            N_modes_descriptors = sample.flow_features.descriptors
            weights = self.weights_predictor.predict(N_modes_descriptors, sample.bc.array())
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