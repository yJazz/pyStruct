import pickle

def pickle_object(object, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(object, f)
    return 
    
def load_pickle(path):
    with open(path, 'rb') as f:
        object = pickle.load(f)
    return object