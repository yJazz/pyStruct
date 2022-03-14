import matplotlib.pyplot as plt
import numpy as np


from dataset import get_training_pairs, sliding_training_pairs, shuffle, split_train_test
from model import make_gru

def random_plot(model, X, y):
    """ Randomly choose the sequence and plot the validation results"""
    case = np.random.randint(low=0, high=len(y))
    y_pred = model.predict(X[case:case+1, :, :]).flatten()
    y_true = y[case,:]
    t = np.arange(len(y_true))

    plot = plt.figure(figsize=(15,3))
    ax1 = plot.add_subplot(121)
    ax1.scatter(t, y_true,s=3, color='r', facecolors='none', label='True')
    ax1.plot(t, y_pred, label='Predict')
    ax1.legend()


def predict_single_seq(model, X_single_slide, y_single_slide):
    y_pred = model.predict(X_single_slide)
    y_pred = y_pred.flatten()
    y_single_slide = y_single_slide.flatten()
    return y_single_slide, y_pred


def plot_training_seq(model, X_single_slide, y_single_slide):
    y_single_slide, y_pred = predict_single_seq(model, X_single_slide, y_single_slide)

    plt.plot(y_single_slide, label='true')
    plt.plot(y_pred, label='predict')
    plt.legend()
    plt.show()   
    return 

def plot_full_history(model, X_slide, y_slide):

    y_pred =  model.predict(X_slide)
    y_pred = y_pred.flatten()
    
    y_slide = y_slide.flatten()
    plt.plot(y_slide, label='true')
    plt.plot(y_pred, label='predict')
    plt.legend()
    plt.show()    
    


if __name__ == '__main__':
    # restore a model file from a specific run by user "vanpelt" in "my-project"
    config = {
              "learning_rate": 0.001,
              "epochs": 10,
              "batch_size": 64,
              "architecture": "gru",
              "sequence_length":100, 
              "sequence_stride":100
           }


    X, y = get_training_pairs()
    X_slide, y_slide = sliding_training_pairs(X, y, sequence_length=100, sequence_stride=100)
    X_slide, y_slide = shuffle(X_slide, y_slide)

    print(f"Shape of data:")
    print(f"X: {X_slide.shape}")
    print(f"y: {y_slide.shape}")

    model = make_gru(X_slide)
    
    model.load_weights(r'F:\project2_phD_bwrx\wandb\run-20220116_221637-215fevmx\files\weights.h5')
    model.summary()
    plot_training_seq(model, X_slide[:1, :, :], y_slide[:1, :])
    


