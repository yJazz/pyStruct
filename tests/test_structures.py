import pytest
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import shutil 
import pandas as pd
from pyStruct.data.datastructures import BoundaryCondition, PodSampleSet, PodSample
from pyStruct.machines.structures import GBLookupStructure, find_corresponding_sample
from pyStruct.machines.errors import LabelNotInStructureTable, StructureModelNoneExist, ModelPathNotFound

@dataclass
class StructureParam:
    x_labels: list[str]
    y_labels: list[str]

@pytest.fixture
def bad_structure_config():
    structure_config = StructureParam(
        x_labels=['m_c', 'v'],
        y_labels=['singular', 'spatial']
        )
    return structure_config

    
class TestGBLookup:
    output_to = Path('./tests/output/test_structure/GB')
    @classmethod
    def setup_class(cls):
        cls.output_to.mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls): 
        shutil.rmtree(cls.output_to)

    def setup_method(self):
        structure_config = StructureParam(
            x_labels=['m_c', 'm_h', 'T_c', 'T_h', 'theta_deg'],
            y_labels=['singular', 'spatial']
            )
        self.machine = GBLookupStructure(
            structure_config, 
            TestGBLookup.output_to)
        return

    def test_1_create_model(self):
        model = self.machine.create_model()
        assert hasattr(model, 'fit')

    def test_2_train(self, samples_with_features):
        self.machine.train(PodSampleSet(samples_with_features))
        
    def test_3_predict_without_model(self, samples):
        
        bc = samples[0].bc
        try:
            self.machine.predict(bc, PodSampleSet(samples))
        except StructureModelNoneExist:
            print("The Model None Exist is captured")
        except:
            assert False

    def test_4_save_but_no_path_ex_given(self):
        # if not call `set_model_path` first
        try:
            self.machine.save()
        except ModelPathNotFound:
            pass
        except:
            assert False

    def test_5_load(self, samples_with_features):
        # if forget to specify the model path
        try:
            self.machine.load()
        except ModelPathNotFound:
            pass
        except:
            assert False
        # Specify the model path
        to_folder = Path(r'.')/'structure_predictor'

        self.machine.train(PodSampleSet(samples_with_features))
        self.machine.save()
        self.machine.load()
        assert len(self.machine._models) != 0
        # clean up
        # shutil.rmtree(to_folder)

    def test_6_predict(self, samples_with_features):
        sample_set = PodSampleSet(samples_with_features)
        self.machine.train(sample_set)

        # predict
        bc = samples_with_features[0].bc
        predicted_descriptors, predicted_samples = self.machine.predict(bc, sample_set)
        assert all([isinstance(prediction, PodSample) for prediction in predicted_samples])
        assert all([ type(prediction) == np.ndarray for prediction in predicted_descriptors])