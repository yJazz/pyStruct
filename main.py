from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from scipy.interpolate import griddata

from pyStruct.config import TwoMachineConfig
from pyStruct.framework import TwoMachineFramework, ValidateFramework
from pyStruct.database.datareaders import dump_pickle, read_pickle, read_csv
from pyStruct.visualizers.plots import plot_cartesian


cs = ConfigStore.instance()
cs.store(name='config', node=TwoMachineConfig)

@hydra.main(config_path="config", config_name='config_3dpod_intercept')
def main(cfg:TwoMachineConfig):
    machine = TwoMachineFramework(cfg)
    machine.train()
    machine.test()

    # Validation
    reconstructor = machine.reconstructor

    val_save_to = Path(cfg.paths.save_to) / 'val_train' 
    val_save_to.mkdir(parents=True, exist_ok=True)
    val = ValidateFramework(machine.training_set, reconstructor)
    val.validate_structure(file = val_save_to /'structure.txt')
    val.validate_optimize(folder = val_save_to)
    val.validate_weights(file = val_save_to/'weights_pred.png')
    val.validate_workflow(folder = val_save_to)

    test_save_to = Path(cfg.paths.save_to) / 'test_train' 
    test_save_to.mkdir(parents=True, exist_ok=True)
    test = ValidateFramework(machine.testing_set, reconstructor)
    test.validate_workflow(folder = test_save_to)

if __name__ == '__main__':
    main()