import json 
from pathlib import Path
import numpy as np
import pandas as pd
import sys


from pyStruct.data.dataset import get_training_pairs, read_pickle, ModesManager
from pyStruct.machines.feature_processors import get_spatial_strength, get_temporal_behavior
from pyStruct.machines.structures import LookupStructure



def get_bcs_from_signac(workspace):
    workspace = Path(workspace)
    state_point_file = workspace/"signac_statepoint.json"
    with open(state_point_file) as f:
        sp = json.load(f)
    
    bcs = {} 
    for sample_name in sp['aggre_sp']:
        bc = sp['aggre_sp'][sample_name]["cfd_sp"]['bc']
        sample = int(sample_name.split("_")[-1])
        bcs[sample] = bc
    return bcs

        

    
class StructurePredictor:
    def __init__(self, wp, x_labels, y_labels):
        self.wp = wp
        self.x_labels = x_labels
        self.y_labels = y_labels

    def train(self):
        sps = []
        modes = self.wp['mode'].unique()
        for mode in modes:
            wp_mode = self.wp[self.wp['mode'] == mode]
            params = {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 1 }
            sp = LookupStructure(wp_mode, self.x_labels, self.y_labels)
            sp.create_model_and_fit(params)
            sps.append(sp)
        self.sps = sps
        return  

    def read(self, file_path):
        """ Read the sps from file"""
        assert file_path.endswith('.pkl'), "Structure should be a pickle"

        sps = read_pickle(file_path) 
        self.sps = sps
        return 

    def compose_baseX_and_features(self, X_lib, input_dict):
        """ Compose X_compose matrix from the library X"""

        print(f"--- finding base modes---")
        # Structure predictors
        sps = self.sps
        X_compose = np.zeros(X_lib.shape[-2:])
        features = pd.DataFrame()
        modes = self.wp['mode'].unique()

        for i, mode in enumerate(modes):
            mode = int(mode)
            # mode result 
            m = sps[mode].get_mode_deterministic_from_library(mode, input_dict)
            sample = int(m['sample'])
            features = pd.concat([features, m], ignore_index = True, axis=1)
            X_compose[i, :] = X_lib[sample, mode, :]
        features = features.transpose()

        return X_compose, features



class TimeSeriesPredictor:
    def __init__(self, config, feature_processor, optimizer, predictor):
        self.config = config

        self.feature_processor = feature_processor
        self.feature_table = feature_processor.feature_table

        self.optimizer = optimizer
        self.predictor = predictor
        self.N_samples = config['N_samples']
        self.N_modes = config['N_modes']
        self.N_t = config['N_t']
        self.feature_names = config['y_labels']
    
    def get_training_pairs(self):
        X, idx = self.feature_processor.get_X()
        y = self.feature_processor.get_y()
        return X, y, idx

    def optimize(self):
        X, y, _ = self.get_training_pairs()
        self.feature_table['w'] = np.zeros(len(self.feature_table))

        # Run optimization and Create table 
        for sample in range(self.config['N_samples']):
            feature_by_sample = self.feature_table[self.feature_table['sample']==sample]
            id = feature_by_sample.index

            print(f"Optimizing case {sample}")
            proc_id = f'proc_{sample}'

            # Get modes in single case
            X_r = X[sample, range(self.config['N_modes']), :]
            y_r = y[sample, ...]
            weights = self.optimizer.optimize(X_r, y_r)

            for i, w in enumerate(weights):
                self.feature_table.loc[id[i], 'w'] = w
        return
    
    def train(self, show=False):
        feature_table = self.predictor.train(self.feature_table, show)
        # self.feature_table = feature_table
        return 

    def predict_weights(self, features):
        weights = self.predictor.predict(features)
        return weights


# class TimeSeriesPredictor_depr:
#     def __init__(self, N_samples, N_modes, N_t, wp):
#         self.N_samples = N_samples
#         self.N_modes = N_modes
#         self.N_t = N_t
#         self.wp = wp
    
#     def get_training_pairs(self, config):
#         pods = ModesManager(name='', workspace=config['workspace'], to_folder=None)
        
#         # y
#         y = pods.T_walls[str(config['theta_deg'])]['T_wall']

#         # X
#         loc_index = pods.T_walls[str(config['theta_deg'])]['loc_index']
#         X_temporals = pods.X_temporals
#         X_s = pods.X_s
#         X_spatials = pods.X_spatials
#         coherent_strength = X_spatials[:, :, :, loc_index]
#         X = np.zeros((pods.N_samples, config['N_modes'], pods.N_t))

#         useful_modes_idx=[]
#         for sample in range(pods.N_samples):
#             cs = coherent_strength[sample, :, :]
#             convert_to_norm = lambda vec: np.linalg.norm(vec) 
#             norm_of_cs = np.apply_along_axis(convert_to_norm, axis=1, arr=cs)
#             for mode in range(pods.N_modes):
#                 norm_of_cs[mode] = norm_of_cs[mode] * X_s[sample, mode]
                
#             args = np.argsort(norm_of_cs)[::-1]
#             useful_modes_idx.append(args)

#             for mode, arg in enumerate(args[:config['N_modes']]):
#                 X[sample, mode, :] = X_temporals[sample, arg, :] * cs[arg, 0]*10
#             # X[sample, mode, :] = X_temporals[sample, mode, :]

#         # return X, y, useful_modes_idx
#         # X = X_temporals
#         # useful_modes_idx = [list(range(20)) for _ in range(20)]
#         return X, y, useful_modes_idx
    
#     def optimize(self, workspace, theta_deg, loss_weights_config=None):
#         """ Take data from the workspace, and optimize the weights at theta_deg"""
#         print("prepare optimize")
#         assert Path(workspace).exists()
#         if loss_weights_config:
#             assert len(loss_weights_config) == 5, "Need to specify: fft, std, min, max, hist"

#         if not loss_weights_config:
#             loss_weights_config = [1, 1, 1, 1, 1]

#         config ={
#             'workspace': workspace,
#             'theta_deg': theta_deg,
#             'N_modes': self.N_modes,
#             'sample_ids': range(self.N_samples),
#             'N_t': self.N_t,
#             "fft_loss_weight":loss_weights_config[0],
#             "std_loss_weight": loss_weights_config[1],
#             "min_loss_weight":loss_weights_config[2],
#             "max_loss_weight":loss_weights_config[3], 
#             "hist_loss_weight":loss_weights_config[4],
#             "maxiter":1000,
#         }

#         X, y, _ = self.get_training_pairs(config)
#         weights_table = optm_workflow(config, X, y)
#         config['weights_table'] = weights_table
#         return config

#     def read_optimize(self, workspace, theta_deg, weights_table, loss_weights_config=None):
#         """ Take data from the workspace, and optimize the weights at theta_deg"""
#         if loss_weights_config:
#             assert len(loss_weights_config) == 5, "Need to specify: fft, std, min, max, hist"

#         if not loss_weights_config:
#             loss_weights_config = [10, 1, 1, 1, 1]
#         config ={
#             'workspace': workspace,
#             'theta_deg': theta_deg,
#             'mode_ids': range(self.N_modes),
#             'sample_ids': range(self.N_samples),
#             'N_t': self.N_t,
#             "fft_loss_weight":loss_weights_config[0],
#             "std_loss_weight": loss_weights_config[1],
#             "min_loss_weight":loss_weights_config[2],
#             "max_loss_weight":loss_weights_config[3], 
#             "hist_loss_weight":loss_weights_config[4],
#             "maxiter":1000,
#         }
#         config['weights_table'] = weights_table
#         return config
    
#     def post_process_wp_table(self, config_optm):
#         # Features
#         optm_weights = config_optm['weights_table']
        
#         # Add weight to the wp
#         for i in range(len(optm_weights)):
#             sample = optm_weights.loc[i, 'sample']
#             for mode in range(self.N_modes):
#             # read weight from weights optimization
#                 wt_s = optm_weights[abs(optm_weights['sample']-sample)<1E-4]
#                 assert len(wt_s) == 1
#                 w_mode = wt_s[f'w{int(mode)}']

#                 id = self.wp[(self.wp['sample']==sample) & (self.wp['mode']==mode) & (self.wp['theta_deg']==config_optm['theta_deg'])].index[0]
#                 self.wp.loc[id, 'w'] = float(w_mode)
#         return 

#     def train(self, config_optms: list, training_id, show=False):
#         """  
#             (1) Take the weights table
#             (2) Compose weight property table
#         """
#         # Post process the wp table
#         self.wp['w'] = np.zeros(len(self.wp)) *np.nan
#         for config_optm in config_optms:
#             self.post_process_wp_table(config_optm)
#         assert self.wp.isnull().values.any() == False, "bad data"

#         # Start training the model
#         params = {
#             "n_estimators": 1000,
#             "max_depth":10,
#             "min_samples_split": 2,
#             "learning_rate": 0.005,
#             "loss": "squared_error",
#         }
#         feature_names = ['mode', 'singular', 'spatial_strength', 'temporal_behavior', 'vel_ratio','theta_deg']
#         # Get weight property tables and
#         wp = self.wp
        
#         wp_train, wp_test = wp_split_train_test(wp, training_id)
#         regs, norms = get_regressors_for_each_mode(
#             self.N_modes,
#             params, 
#             wp_train, 
#             feature_names, 
#             train_ratio=0.8, 
#             show=show)         

#         self.wp_train = wp_train
#         self.wp_test = wp_test
#         self.regs = regs
#         self.norms = norms
#         self.workspace = config_optms[0]['workspace']
#         self.feature_names = feature_names
#         return 

#     def predict_weights(self, features):
#         assert hasattr(self, 'regs'), "No regressors exist. Train or Read first"
#         assert features.shape == (self.N_modes, len(self.feature_names))
            
#         weights=[]
#         for mode in range(self.N_modes):
#             reg = self.regs[mode]
#             norm = self.norms[mode]
#             # normalize the feature
#             normalized_feature_mode = norm.transform(features)[mode, :]
#             weights.append(reg.predict(normalized_feature_mode.reshape(1,len(self.feature_names))))
#         return weights


class TwoMachineFramework:
    def __init__(self, structure_predictor, time_series_predictor):
        self.stcr = structure_predictor
        self.ts = time_series_predictor

        self.N_samples = self.ts.N_samples
        self.N_modes = self.ts.N_modes
        self.N_t = self.ts.N_t

    def compare_predict(self, sample):
        X, y, _ = self.ts.get_training_pairs()
        y_true = y[sample, ...]

        # Structures are given 
        feature_table = self.ts.feature_table
        ft_f = feature_table[(feature_table['sample']==sample) ]

        features_modes = ft_f[(ft_f['mode'].isin(list(range(self.N_modes))) )]


        # Time-series predictors
        features = features_modes[self.ts.feature_names]
        weights = self.ts.predict_weights(features)
        print(weights)
        y_pred = np.array(
            [X[sample, mode, :] * weights[mode] for mode in range(self.ts.N_modes) ] ).sum(axis=0)
        return y_true[-self.N_t:], y_pred


    def predict(self, input_dict: dict):
        """ Given xlabels, output time-series"""
        # Read X, y
        X, _, _ = self.ts.get_training_pairs()

        # Structuer predictors: predict structures 
        X_compose, features = self.stcr.compose_baseX_and_features(X, input_dict)

        # Time Series Predictor
        # Transform the structure predictor resutls 
        features = features[self.ts.feature_names]
        weights = self.ts.predict_weights(features)
        print(weights)
        y_pred = np.array(
            [X_compose[mode, :] * weights[mode] for mode in range(self.ts.N_modes) ] ).sum(axis=0)
        return y_pred

