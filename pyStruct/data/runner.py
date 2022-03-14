import datetime
import os
import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback

from utils import get_psd



def training_depr(model, X, Y, folder, batch_size=12):
    step = 5E-4

    # Compile MOdel 
    optimizer = tf.keras.optimizers.Adam(step)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.metrics.MeanSquaredError()
    model.compile(optimizer, mse_loss_fn, loss_metric, run_eagerly=True)

    # Set up 
    log_dir = "%s/"%folder + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      mode='min')
    
    checkpoint_path = os.path.join(log_dir, 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=5)


    model.fit(X, Y, 
              epochs= 100, 
              batch_size=batch_size,
              validation_split=0.2,
              validation_batch_size=batch_size,
              shuffle=True,
              callbacks=[tensorboard_callback, early_stopping, cp_callback]
             )
    return 

def fft_loss(y, y_pred):
    y = tf.cast(y, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)
    fft_y = tf.signal.fft(y)
    fft_y_pred = tf.signal.fft(y_pred)
    
    # Find the distance between the two vectors 
    loss = tf.norm(fft_y - fft_y_pred)
    loss = tf.cast(loss, tf.float32)
    return loss
    

def psd_loss(y, y_pred):
    y = tf.cast(y, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)
    fft_y = tf.signal.fft(y)
    fft_y_pred = tf.signal.fft(y_pred)
    
    # Find the distance between the two vectors 
    loss = tf.math.real(tf.norm(fft_y - fft_y_pred))
    loss = tf.cast(loss, tf.float32)
    return loss

def std_loss(y, y_pred):
    y = tf.cast(y, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    std_true = tf.abs(tf.math.reduce_std(y))
    std_pred = tf.abs(tf.math.reduce_std(y_pred))
    loss = tf.abs(tf.math.subtract(std_true, std_pred))
    loss = tf.cast(loss, tf.float32)
    return loss


def psd_and_std_loss(y, y_pred, std_weight=0.5):
    """ compute the time-series loss based on:
        1. FFT: compare the distribution
        2. std
    """

    loss_psd = psd_loss(y, y_pred)
    loss_std = std_loss(y, y_pred)
    print(f"loss: psd {loss_psd}, std {loss_std}")
    # print(f'loss_std: {loss_std}')
    total_loss = tf.math.add(loss_psd, std_weight*loss_std)

    return total_loss








def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred)
    # print(f'----- Loss value: {loss_value}')
    # print(f"Trainable variables: {model.trainable_weights} \n")

    grads = tape.gradient(loss_value, model.trainable_weights)
    # print(f"---- Gradients: type={type(grads)}, value: {grads}\n ")
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, y_pred)
    return loss_value

def test_step(x, y, model, loss_fn, val_acc_metric):
    y_val = model(x, training=False)
    loss_value = loss_fn(y, y_val)
    val_acc_metric.update_state(y, y_val)
    return loss_value




def training(model, train_dataset, val_dataset, batch_size=12, epochs=10, 
             loss_fn = tf.keras.losses.MeanSquaredError(), step=1E-3
             ):

    # Compile MOdel 
    optimizer = tf.keras.optimizers.Adam(step)
    train_acc_metric = tf.metrics.MeanSquaredError()
    val_acc_metric = tf.metrics.MeanSquaredError()
    model.compile(optimizer, loss_fn, train_acc_metric, run_eagerly=True)

    # make dataset and do batching
    

    for epoch in range(epochs):
        print(f"\n Start of epoch {epoch}")

        train_loss = []
        val_loss = []

        # Iterate
        
         # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)

            train_loss.append(float(loss_value))
        print(f'model weights: {model.trainable_weights[0].numpy()[0,0]}')
        print(f"Training loss: {loss_value}")

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
        
        # Save a model file from the current directory
        model.save(os.path.join(wandb.run.dir, 'model.h5'))
        model.save_weights(os.path.join(wandb.run.dir, 'weights.h5'))

    return model