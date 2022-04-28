from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from pyStruct.config import TwoMachineConfig


class TestConfig:
    def setup(self):
        with initialize(config_path='test_data', job_name='test'):
            cs = ConfigStore.instance()
            cs.store(name='config_pod', node=TwoMachineConfig)

            cfg = compose(config_name='config_pod')
            self.cfg = cfg
        return 
    def test_compose(self):
        print(self.cfg)

    def test_convert_bcmapper_to_dict(self):
        bc_mapper = self.cfg.signac_sample.bc_mapper
        assert type(dict(bc_mapper)) == dict