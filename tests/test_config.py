import hydra
from hydra import compose, initialize
from hydra.errors import MissingConfigException
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import dataclasses

from pyStruct.config import TwoMachineConfig
from pyStruct.machines.feature_processors import FeatureProcessor


def test_1_load_config() -> None:
    with initialize(config_path='test_data', job_name='test'):
        cfg = compose(config_name='config_good')
        assert set(cfg.keys()) == set([field.name for field in dataclasses.fields(TwoMachineConfig)])

def test_2_import_machine_config() -> None:
    """ Make sure: every line in the yamal file belongs to a dataclass"""

    # A good yaml file
    with initialize(config_path='test_data', job_name='test'):
        cs = ConfigStore.instance()
        cs.store(name='config_good', node=TwoMachineConfig)

        cfg = compose(config_name='config_good')

    # A bad yaml file
    with initialize(config_path='test_data', job_name='test'):
        cs = ConfigStore.instance()
        cs.store(name='two_machine_config', node=TwoMachineConfig)
        cfg_bad = compose(config_name='config_bad', return_hydra_config=True)
        print(cfg_bad.keys())
