import numpy as np
import matplotlib.pyplot as plt


class TwoMachineFramework:
    def __init__(
        self,
        config: dict,
        feature_processor,
        optimizer, 
        weights_predictor, 
        structure_predictor
    ):
        # Assertions: Make sure each inputs are of the right class
        self.config = config
        self.feature_processor = feature_processor(config)
        self._init_data()
        self.optimizer = optimizer(config)
        self.weights_predictor = weights_predictor(config)
        self.structure_predictor = structure_predictor(config)

    def _init_data(self):
        self.feature_table = self.feature_processor.feature_table

    def train(self, show=False):
        # Weight optimization 
        print(f'=========== Optimization ==========')
        self._optimize()

        # Time-series Predictor: the models are saved in the weigths_predictors
        # check weights_predictors
        print(f'=========== Weights ==========')
        self.weights_predictor.train(self.feature_table)

        # Structure Predictor
        print(f'=========== Structures ==========')
        self.structure_predictor.train(self.feature_table)
    
    def predict(self, input_dict: dict):
        assert list(input_dict.keys())  == self.config['x_labels']

        # Retrieve base X: 
        X, _ = self.feature_processor.get_training_pairs()


        # Predict structures
        X_compose, features  = self.structure_predictor.predict_and_compose(input_dict, X)

        # Predict weights
        weights = self.weights_predictor.predict(features)

        # Output
        y_pred = np.array(
            [X_compose[mode, :] * weights[mode] for mode in range(self.config['N_modes']) ] ).sum(axis=0)
        return y_pred
    
    def validation(self, plot=True):
        """ Visualize the training samples """
        # Get reference data
        X, y_trues = self.feature_processor.get_training_pairs()

        y_preds = []
        # Get boundary conditions
        feature_table = self.feature_processor.feature_table
        for sample in range(self.config['N_samples']):
            f = feature_table[feature_table['sample'] == sample]

            input = {x_label: f[x_label].values[0] for x_label in self.config['x_labels']}
            y_pred = self.predict(input)
            y_preds.append(y_pred)

        if plot:
            ncols=5
            nrows = int(np.ceil(self.config['N_samples']/ncols))
            figsize = (ncols*3.2, nrows*3)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            for sample in range(self.config['N_samples']):
                ax = axes[sample//ncols, sample%ncols]
                ax.plot(y_trues[sample, :], label='True')
                ax.plot(y_preds[sample], label='Pred')

            ax.legend(bbox_to_anchor=(1.1, 0))
            plt.tight_layout()
            plt.show()

        return y_trues, y_preds

    def _optimize(self):
        # Retrieve data
        X, y = self.feature_processor.get_training_pairs()

        # Create Space holder
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

            # Call the optimizer
            weights = self.optimizer.optimize(X_r, y_r)

            for i, w in enumerate(weights):
                self.feature_table.loc[id[i], 'w'] = w
        return