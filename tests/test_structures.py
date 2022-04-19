from pathlib import Path
import shutil 
import pandas as pd
from pyStruct.machines.datastructures import BoundaryCondition
from pyStruct.machines.structures import GBLookupStructure, find_corresponding_mode
from pyStruct.machines.errors import LabelNotInStructureTable, StructureModelNoneExist, ModelPathNotFound


# def test_1_create_model(good_structure_config):
#     structure_library = pd.read_csv('tests/test_data/structure_table.csv')
#     gb_structure = GBLookupStructure(good_structure_config, structure_library)
#     model = gb_structure.create_model()
#     assert hasattr(model, 'fit')

# def test_2_check_labels(bad_structure_config):
#     structure_library = pd.read_csv('tests/test_data/structure_table.csv')
#     machine = GBLookupStructure(bad_structure_config, structure_library)
#     try:
#         machine._check_labels(bad_structure_config.x_labels, bad_structure_config.y_labels, structure_library)
#     except LabelNotInStructureTable:
#         print("check label is captured")
#     except:
#         assert False

# def test_3_train(good_structure_config):
#     structure_library = pd.read_csv('tests/test_data/structure_table.csv')
#     machine = GBLookupStructure(good_structure_config, structure_library)
#     machine.train(structure_library)
    
# def test_4_predict_without_model(good_structure_config):
#     structure_library = pd.read_csv('tests/test_data/structure_table.csv')
#     machine = GBLookupStructure(good_structure_config, structure_library)
#     m_c = structure_library.loc[0, 'm_c']
#     m_h = structure_library.loc[0, 'm_h']
#     T_c = structure_library.loc[0, 'T_c']
#     T_h = structure_library.loc[0, 'T_h']
#     bc = BoundaryCondition(m_c=m_c, m_h=m_h, T_c = T_c, T_h = T_h)
#     try:
#         machine.predict(bc)
#     except StructureModelNoneExist:
#         print("The Model None Exist is captured")
#     except:
#         assert False

# def test_5_save(good_structure_config):
#     structure_library = pd.read_csv('tests/test_data/structure_table.csv')
#     machine = GBLookupStructure(good_structure_config, structure_library)
#     # if not call `set_model_path` first
#     try:
#         machine.save()
#     except ModelPathNotFound:
#         pass
#     except:
#         assert False

def test_6_load(good_structure_config):
    structure_library = pd.read_csv('tests/test_data/structure_table.csv')
    machine = GBLookupStructure(good_structure_config, structure_library)
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
    machine.train()
    machine.save()
    machine.load()
    assert len(machine._models) != 0
    # clean up
    # shutil.rmtree(to_folder)

    # machine.set_model_path(to_folder)
    # machine.train()
    # predict
    m_c = structure_library.loc[0, 'm_c']
    m_h = structure_library.loc[0, 'm_h']
    T_c = structure_library.loc[0, 'T_c']
    T_h = structure_library.loc[0, 'T_h']
    bc = BoundaryCondition(m_c=m_c, m_h=m_h, T_c = T_c, T_h = T_h)
    predictions = machine.predict(bc)

def test_7_validate(good_structure_config):
    structure_library = pd.read_csv('tests/test_data/structure_table.csv')
    machine = GBLookupStructure(good_structure_config, structure_library)
    machine._validate_structure()

