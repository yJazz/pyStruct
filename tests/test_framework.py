from pathlib import Path
from framework import *
import pickle

from structure_predictor import structures
from data.dataset import get_training_pairs

config = {
    # Wandb
    'wandb_record': False, 
    'wandb_project_name':'two_machine_framework',
    'wandb_run_name': 'training',

    # Signac
    'signac_aggre_folder': r"F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307",
    
    # Regression data
    'reg_data_folder':r"F:\project2_phD_bwrx\prediction_data",

    # Structure prediction
    'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
    'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
    'train_id': range(15) ,
    'theta_degs': [0, 5, 10, 15, 20, 25, 30, 45], 
}


def setup_wp_table():
    # Get the weights property table
    wp_table_path = r"F:\project2_phD_bwrx\prediction_data\wp_0_5_10_15_20_25_30_45.csv"
    assert Path(wp_table_path).exists()
    wp =  pd.read_csv(wp_table_path)
    return wp

def setup_sps():
    wp = setup_wp_table()
    sps = []
    for mode in range(20):
        wp_mode = wp[wp['mode'] == mode]
        params = {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 1 }
        sp = structures.LookupStructure(wp_mode, config['x_labels'], config['y_labels'])
        sp.create_model_and_fit(params)
        sps.append(sp)
    return sps

def setup_input_dict():
    input_dict = {
        'm_c': 400,
        'vel_ratio':100,
        'theta_deg':0
    }
    return input_dict

def setup_library():
    workspace = Path(r"F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307")
    X_lib, y_lib = get_training_pairs(workspace, theta_deg=0)
    return X_lib, y_lib

# =========================================================================
def test1_initialize_wp_table():
    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }

    wp = initialize_wp_table(config)
    print(wp)

def test1_strucure_predictors():
    wp = setup_wp_table()
    x_labels = ['m_c', 'vel_ratio', 'theta_deg']
    y_labels= ['spatial_strength', 'temporal_behavior', 'singular']
    strc = StructurePredictor(wp, x_labels, y_labels)
    strc.train()

    sps = strc.sps
    mode = 0
    input_dict = setup_input_dict()
    m = sps[mode].get_mode_deterministic_from_library(mode, input_dict)
    assert type(m) == pd.Series
    assert all([input in m.keys() for input in config['x_labels']])
    
def test2_get_training_pairs():
    signac_workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    # Read 
    X_temporal_path = Path(signac_workspace)/'features'/ 'X_temporals.pkl'
    y_theta_0_path = Path(signac_workspace)/ 'target'/ 'target_deg0_loc1584.pkl'
    with open(X_temporal_path, 'rb') as f:
        X_real = pickle.load(f)
    with open(y_theta_0_path, 'rb') as f:
        y_real = pickle.load(f)
        y_real = y_real - y_real.mean(axis=1).reshape(-1,1)
    
    # Read from my module 
    X, y = get_training_pairs(workspace=signac_workspace, theta_deg=0)
    assert (X == X_real).all()
    assert (y == y_real).all()


def test4_ts_optmization():

    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }

    wp = initialize_wp_table(config)
    ts = TimeSeriesPredictor(N_samples=2, N_modes=2, N_t=1000, wp=wp)
    workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    optimized_result = ts.optimize(workspace, theta_deg=0)
    print(optimized_result['weights_table'])
    from optimization import reconstruct_optimize
    y, y_preds = reconstruct_optimize(optimized_result)    
    assert y.shape == y_preds.shape

def test5_ts_post_process_wp_table():
    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }

    wp = initialize_wp_table(config)
    ts = TimeSeriesPredictor(N_samples=2, N_modes=2, N_t=1000, wp=wp)
    workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    optimized_result = ts.optimize(workspace, theta_deg=0)
    ts.post_process_wp_table(config_optm = optimized_result)
    wp_table = ts.wp
    assert 'spatial_strength' in wp_table.keys()
    assert 'temporal_behavior' in wp_table.keys()
    assert 'vel_ratio' in wp_table.keys()

def test6_ts_train():
    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }

    wp = initialize_wp_table(config)
    ts = TimeSeriesPredictor(N_samples=2, N_modes=2, N_t=1000, wp=wp)
    workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    config_optms = [ ts.optimize(workspace, theta_deg) for theta_deg in [0, 5, 15]]
    ts.train(config_optms, training_id=range(15))
    assert hasattr(ts, 'regs')
    assert type(ts.wp_train) == pd.DataFrame
    assert len(ts.regs) == 2
    assert 'predict' in dir(ts.regs[0])

def test7_ts_predict_weights():
    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }

    wp = initialize_wp_table(config)
    ts = TimeSeriesPredictor(N_samples=2, N_modes=2, N_t=1000, wp=wp)
    workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    config_optms = [ ts.optimize(workspace, theta_deg) for theta_deg in [0, 5, 15]]
    ts.train(config_optms, training_id=range(15))

    # print(wp.head())
    feature_names = ['mode', 'singular', 'spatial_strength', 'temporal_behavior', 'vel_ratio']
    features = wp.loc[0:1, feature_names]
    weights = ts.predict_weights(features)
    print(f'weights:{weights}')


def test8_TwoMachine():
    config = {
        'workspace':r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        'N_samples':2,
        'N_modes':2,
        'N_t':1000,
        'theta_degs':[0, 5, 10],
        'training_id':range(15),
        'x_labels':['m_c', 'vel_ratio', 'theta_deg'],
        'y_labels':['spatial_strength', 'temporal_behavior', 'singular'],
        'save_to': Path(r'F:\project2_phD_bwrx\prediction_data\20220224_linear')
    }
    wp = initialize_wp_table(config)
    # Structure Predictor
    x_labels = ['m_c', 'vel_ratio', 'theta_deg']
    y_labels= ['spatial_strength', 'temporal_behavior', 'singular']
    strc = StructurePredictor(wp, x_labels, y_labels)
    strc.train()

    # Time-series Predictor
    ts = TimeSeriesPredictor(N_samples=2, N_modes=2, N_t=1000, wp=wp)
    workspace = r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307'
    config_optms = [ ts.optimize(workspace, theta_deg) for theta_deg in [0, 5, 15]]
    ts.train(config_optms, training_id=range(15))


    # Construct TwoMachine
    machine = TwoMachineFramework(strc, ts)
    input_dict = setup_input_dict()
    machine.predict(input_dict)



# def test3_Two_MachineFrameWork_compose_baseX_and_features():
#     sps = setup_sps()
#     input_dict = setup_input_dict()
#     X_lib, y_lib = setup_library()
#     stcr = StructurePredictor(sps)
#     ts = TimeSeriesPredictor()
#     N_samples, N_modes, N_t = X_lib.shape
#     machine = TwoMachineFramework(stcr, ts, N_samples, N_modes, N_t)
#     X_compose, features = machine.compose_baseX_and_features(X_lib, input_dict)
#     assert X_compose.shape== (N_modes, N_t)
#     assert type(features) == pd.DataFrame

if __name__ == "__main__":
    test5_ts_post_process_wp_table()

