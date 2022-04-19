import pytest
from pathlib import Path
import numpy as np
import shutil 
import pandas as pd
from pyStruct.machines.datastructures import BoundaryCondition, PodSampleSet, PodSample
from pyStruct.machines.structures import GBLookupStructure, find_corresponding_sample
from pyStruct.machines.errors import LabelNotInStructureTable, StructureModelNoneExist, ModelPathNotFound

@pytest.fixture
def samples_with_features(samples):
    for sample in samples:
        sample.set_flowfeatures(time_series=np.random.rand(2,10), descriptors=np.random.rand(2,4))
    return samples

def test_1_create_model(good_structure_config):
    gb_structure = GBLookupStructure(good_structure_config)
    model = gb_structure.create_model()
    assert hasattr(model, 'fit')

def test_2_train(good_structure_config, samples_with_features):
    machine = GBLookupStructure(good_structure_config)
    machine.train(PodSampleSet(samples_with_features))
    
def test_3_predict_without_model(good_structure_config, samples):
    
    machine = GBLookupStructure(good_structure_config)
    bc = samples[0].bc
    try:
        machine.predict(bc, PodSampleSet(samples))
    except StructureModelNoneExist:
        print("The Model None Exist is captured")
    except:
        assert False

def test_4_save_but_no_path_ex_given(good_structure_config):
    machine = GBLookupStructure(good_structure_config)
    # if not call `set_model_path` first
    try:
        machine.save()
    except ModelPathNotFound:
        pass
    except:
        assert False

def test_5_load(good_structure_config, samples_with_features):
    machine = GBLookupStructure(good_structure_config)
    # if forget to specify the model path
    try:
        machine.load()
    except ModelPathNotFound:
        pass
    except:
        assert False
    # Specify the model path
    to_folder = Path(r'.')/'structure_predictor'
    machine.set_model_path(to_folder)

    machine.train(PodSampleSet(samples_with_features))
    machine.save()
    machine.load()
    assert len(machine._models) != 0
    # clean up
    # shutil.rmtree(to_folder)

def test_6_predict(good_structure_config, samples_with_features):
    machine = GBLookupStructure(good_structure_config)

    to_folder = Path(r'.')/'structure_predictor'
    machine.set_model_path(to_folder)

    sample_set = PodSampleSet(samples_with_features)
    machine.train(sample_set)

    # predict
    bc = samples_with_features[0].bc
    predictions = machine.predict(bc, sample_set)
    assert all([isinstance(prediction, PodSample) for prediction in predictions])