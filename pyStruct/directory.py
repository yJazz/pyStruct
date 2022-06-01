from dataclasses import dataclass
from pathlib import Path


class FrameworkPaths:
    """ 
    Create the machine directories
    """
    def __init__(self, root, machine_config, create_folder=True):
        Path(root).mkdir(parents=True, exist_ok=True)
        print(" The model is saved at: ")
        for key, value in machine_config.items():
            machine_path = Path(root)/key
            machine_path.mkdir(parents=True, exist_ok=True)

            model_path = machine_path/value
            model_path.mkdir(parents=True, exist_ok=True)
            setattr(self, key, model_path)
            print(f'{key}: {model_path}')

