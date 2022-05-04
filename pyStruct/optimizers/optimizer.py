from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        raise NotImplementedError

    @abstractmethod
    def set_all_weights(self, all_weights: np.ndarray):
        raise NotImplementedError


def update_optimization_record(all_weights:np.ndarray, sample_names:np.ndarray, save_to):
    record = pd.DataFrame(all_weights, columns=[f'w{mode}' for mode in range(all_weights.shape[1])])
    record.insert(0, 'sample', sample_names)
    record.to_csv(save_to, index=False)
    return

def optimize_sample(optimizer, sample):
    print(f'Optimize sample: {sample.name}')
    X = sample.flow_features.time_series
    y = sample.wall_true
    weights = optimizer.optimize(X, y) # shape: (N_modes, )
    return weights


def optimize(optimizer, sample_set) -> np.ndarray:
    record_file = optimizer.save_to/'optm.csv'
    assert optimizer.save_to.exists()
    try: 
        record = pd.read_csv(record_file, index_col='sample')
    except FileNotFoundError:
        print("Optimization not found; start optimization")

    all_weights = []
    for sample in sample_set.samples:
        try:
            weights = record.loc[sample.name].values
            print(f'load sample: {sample.name}')
        except KeyError:
            print(f'New sample to optimize: {sample.name}')
            weights = optimize_sample(optimizer, sample)
        except NameError:
            print(f'New sample to optimize: {sample.name}')
            weights = optimize_sample(optimizer, sample)
        except:
            raise Exception(" Something's wrong")
        all_weights.append(weights)
    
    # update record
    all_weights = np.array(all_weights)
    update_optimization_record(
        all_weights, 
        sample_names= sample_set.name,
        save_to=record_file)
    
    optimizer.set_all_weights(all_weights)
    return 
        
