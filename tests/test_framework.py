import pytest
from pathlib import Path
import shutil
import hydra

import sys
sys.path.insert(0,r'F:\project2_phD_bwrx\code\pyStruct')


from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from pyStruct.config import TwoMachineConfig
from pyStruct.framework import TwoMachineFramework
from pyStruct.featureProcessors.featureProcessors import FeatureProcessor
from pyStruct.optimizers.optimizer import Optimizer
from pyStruct.weightsPredictor.regressors import WeightsPredictor
from pyStruct.structurePredictor.structures import StructurePredictorInterface
from pyStruct.reconstructors import ReconstructorInterface

class TestPodFramework:
    @classmethod
    def setup_class(cls):
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_pod', node=TwoMachineConfig)
            cfg = compose(config_name='config_pod')
        Path(cfg.paths.save_to).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def teardown_class(cls): 
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_pod', node=TwoMachineConfig)
            cfg = compose(config_name='config_pod')
        shutil.rmtree(cfg.paths.save_to)

    def setup_method(self):
        """ instantiate a new machine """
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_pod', node=TwoMachineConfig)
            cfg = compose(config_name='config_pod')
        machine = TwoMachineFramework(cfg)
        self.machine = machine

    def teardown_method(self):
        """ Clear all data. Make every test independent"""
        try:
            shutil.rmtree(self.machine.paths.structure_predictor / 'model.pkl')
        except:
            pass

    def test_1_check_submachines(self):
        assert issubclass(type(self.machine.feature_processor), FeatureProcessor)
        assert issubclass(type(self.machine.structure_predictor), StructurePredictorInterface)
        assert issubclass(type(self.machine.optimizer), Optimizer)
        assert issubclass(type(self.machine.weights_predictor), WeightsPredictor)
        assert issubclass(type(self.machine.reconstructor), ReconstructorInterface)

    def test_2_train_structure_new_train(self):
        self.machine._train_structure()

    def test_3_train_structure_load(self):
        # Load
        self.machine._train_structure()   
        self.machine._train_structure()

    def test_4_optimize(self):
        config = self.machine.config
        N_modes = config.features.N_modes
        assert all( [hasattr(sample, 'w_optm') is False for sample in self.machine.training_set.samples])
        self.machine._optimize()
        assert all( [hasattr(sample, 'w_optm') for sample in self.machine.training_set.samples])
        assert self.machine.training_set.w_optm.shape == (len(self.machine.training_set.samples), N_modes)

    def test_5_train_and_predict(self):
        self.machine.train()
        sample = self.machine.training_set.samples[0]
        self.machine.predict(sample.bc)

class TestDmdFramework:
    @classmethod
    def setup_class(cls):
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_dmd', node=TwoMachineConfig)
            cfg = compose(config_name='config_dmd')
        Path(cfg.paths.save_to).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def teardown_class(cls): 
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_dmd', node=TwoMachineConfig)
            cfg = compose(config_name='config_dmd')
        shutil.rmtree(cfg.paths.save_to)

    def setup_method(self):
        """ instantiate a new machine """
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_dmd', node=TwoMachineConfig)
            cfg = compose(config_name='config_dmd')
        machine = TwoMachineFramework(cfg)
        self.machine = machine

    def teardown_method(self):
        """ Clear all data. Make every test independent"""
        try:
            shutil.rmtree(self.machine.paths.structure_predictor / 'model.pkl')
        except:
            pass

    def test_1_check_submachines(self):
        assert issubclass(type(self.machine.feature_processor), FeatureProcessor)
        assert issubclass(type(self.machine.structure_predictor), StructurePredictorInterface)
        assert issubclass(type(self.machine.optimizer), Optimizer)
        assert issubclass(type(self.machine.weights_predictor), WeightsPredictor)
        assert issubclass(type(self.machine.reconstructor), ReconstructorInterface)

    def test_2_train_structure_new_train(self):
        self.machine._train_structure()

    def test_3_train_structure_load(self):
        # Load
        self.machine._train_structure()   
        self.machine._train_structure()

    def test_4_optimize(self):
        config = self.machine.config
        N_modes = config.features.N_modes
        assert all( [hasattr(sample, 'w_optm') is False for sample in self.machine.training_set.samples])
        self.machine._optimize()
        assert all( [hasattr(sample, 'w_optm') for sample in self.machine.training_set.samples])
        assert self.machine.training_set.w_optm.shape == (len(self.machine.training_set.samples), N_modes)

    def test_5_train_and_predict(self):
        self.machine.train()
        sample = self.machine.training_set.samples[0]
        self.machine.predict(sample.bc)
