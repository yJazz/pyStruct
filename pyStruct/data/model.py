import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional



def make_gru(X_train):
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=50, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=100))
    # Compiling the RNN
    return regressorGRU

def make_linear(X_train):
    N_w, N_m, N_t = X_train.shape
    regressor = Sequential()
    regressor.add(MixModes(N_m, name='mix'))
    return regressor


class MixModes(tf.keras.layers.Layer):
    def __init__(self, N_modes, name=None, **kwargs):
        super(MixModes, self).__init__(name=name)
        self.N_modes = N_modes
        super(MixModes, self).__init__(**kwargs)
        
    def build(self, input_shape):
        initial_value =  tf.ones((1, self.N_modes)) * 0.002
        self.w = tf.Variable(name='mode_weight', shape=(1,self.N_modes), initial_value=initial_value, dtype='float32', trainable=True)
    
    def call(self, modes):
        return tf.matmul(tf.abs(self.w), modes)

    def get_config(self):
        config = super(MixModes, self).get_config()
        config.update({
            'N_modes': self.N_modes
        })
        return config

class MeanLevel_v2(tf.keras.layers.Layer):
    def __init__(self):
        super(MeanLevel_v2, self).__init__()
        
    def build(self, input_shape):
        
        self.c = tf.Variable(name='c', initial_value=0.1, dtype='float32', trainable=True)
    def call(self, T_h, T_c):
        return  self.c*(T_h - T_c) + T_c

class PercentageModel(tf.keras.Model):
    def __init__(self, N_modes):
        super(PercentageModel, self).__init__()
        self.amp_loss_fn =  tf.keras.losses.MeanSquaredError()
        self.freq_loss_fn =  tf.keras.losses.KLDivergence()
        self.mean_loss_fn =  tf.keras.losses.MeanSquaredError()
        self.N_modes = N_modes
        self.fluc_model = MixModes(N_modes)
        # self.meanlevel = MeanLevel_v2()
        
    def call(self, inputs):
        X_temporal = inputs[:, :self.N_modes, :]
        # T_h = inputs[:, self.N_modes, -1]
        # T_c = inputs[:, self.N_modes+1, -1]
        
        fluc = self.fluc_model(X_temporal)
        fluc = tf.math.reduce_sum(fluc, axis=1)
        mean = self.meanlevel(T_h, T_c)
        return mean, fluc
    
    def predict(self, inputs):    
        mean, fluc = self(inputs)        
        return fluc 
    
    def get_amp(self, signal):
        std = tf.math.reduce_std(signal, axis=1)
        return std
    
    def get_mean(self, signal):
        return tf.math.reduce_mean(signal, axis=1)
    
    def get_fluc(self, signal):
        mean = tf.math.reduce_mean(signal, axis=1)
        return tf.convert_to_tensor([signal[i, :] - mean[i] for i in  range( signal.shape[0])])
    
    def get_fft(self, signal):        
        psds = []
        for i in range(signal.shape[0]):
            x = signal.numpy()[i, :]
            f, psdx = get_psd(dt_s=0.005, x=x)
            psds.append(psdx)        
        return tf.convert_to_tensor(psds, dtype=float)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            
            # Predict
            mean_pred, fluc_pred = self(x, training=True)

            # True
            fluc_t = self.get_fluc(y)
            mean_t = self.get_mean(y)
            
            print(f'fluc_t: {fluc_t[0, :5]}')
            print(f'fluc_pred: {fluc_pred[0, :5]}')

            l_amp = self.compiled_loss(self.get_amp(fluc_t), self.get_amp(fluc_pred))
            print(f'freq loss: {l_f}')
            l_mean = self.compiled_loss(mean_t, mean_pred)
            print(f'mean loss: {l_mean}')            
            
            l_f = self.freq_loss_fn(self.get_fft(fluc_t), self.get_fft(fluc_pred))                        
    
            loss =  l_amp + l_f + l_mean
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        output = {m.name: m.result() for m in self.metrics}
        return output
            