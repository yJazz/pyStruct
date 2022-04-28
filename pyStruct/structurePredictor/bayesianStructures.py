
class BayesianLookupStructure(GeneralPredictor):
    def __init__(
        self, 
        config:dict,
        ):

        import arviz as az 
        import pymc3 as pm
        import scipy.stats as stats
        from theano import shared
        from sklearn.preprocessing import StandardScaler
        self.config = config
        self.var_names = config['x_labels']
        self.target_name = 'singular'

        self.save_folder = Path(config['save_to'])/f"bayesian_singular"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_as = self.save_folder/"idata.nc"

    def _init_data(self, feature_table):
        # get data
        self.feature_table = feature_table
        self.target, self.norm_target = self.transform_input(self.target_name)
        self.inputs = {}
        self.inputs_shared={}
        self.standrdizers = {}
        for var_name in self.var_names:
            input, standrdizer = self.transform_input(var_name)
            self.inputs_shared[var_name] = shared(input) # the shared tensor, which can be modified later
            self.standrdizers[var_name] = standrdizer
            self.inputs[var_name] = input # this won't change

    def build_model(self):
        with pm.Model() as model:
            # Hyperpriors for group nodes
            eps = pm.HalfCauchy("eps", 4)
            a = pm.Normal('a', mu=1, sd=1)
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = pm.Normal(var_name, mu = 0, sd= 10 )
                mu = a + b_i * self.inputs_shared[var_name]
            w = pm.Normal( "w", mu=mu, sigma=eps, observed=self.target)
        self.model = model
        return

    def transform_input(self, var_name):
        standardizer = StandardScaler()
        var = standardizer.fit_transform(self.feature_table[var_name].values.reshape(-1,1)).flatten()
        return var, standardizer
    
    def train(self, feature_table: pd.DataFrame, show=False, samples=1000):
        """ Inference """
        self._init_data(feature_table)
        self.build_model()
        with self.model:
            # step = pm.NUTS(target_accept=0.95)
            idata = pm.sample(samples, tune=1000, return_inferencedata=True, chains=1)
        self.idata = idata
        idata.to_netcdf(self.save_as)
        with open(self.save_folder / "standardizer.pkl", 'wb') as f:
            pickle.dump(self.standrdizers, f)
        return 

    def load(self, feature_table):
        print("load Bayesian weights predictor")
        self._init_data(feature_table)
        self.build_model()
        self.idata = az.from_netcdf(self.save_folder/"idata.nc")
        with open(self.save_folder / "standardizer.pkl", 'rb') as f:
            self.standrdizers = pickle.load(f)
        return

    def _predict(self, inputs: dict):
        """ Give the features, """

        # Get the model parameters
        df = az.summary(self.idata)
        # the mean value
        a = df.loc['a', 'mean']

        pred = a
        for i, var_name in enumerate(self.var_names):
            b_i = df.loc[var_name, 'mean']
            pred += b_i*self.standrdizers[var_name].transform(np.array(inputs[var_name]).reshape(-1, 1))
        # Rescale 
        pred = self.norm_target.inverse_transform(pred)

        # Finde the modes 
        idx = []
        for mode in range(self.config['N_modes']):
            # todo 
            pass

        return 

    def predict_and_compose(self, inputs: dict, X_base: np.ndarray, perturb_structure: bool):
        X_compose = np.zeros(X_base.shape[1:])
        predicted_features = self._predict(inputs)

        # Compose 
        for i in predicted_df.index:
            sample = predicted_df.loc[i, 'sample']
            mode = predicted_df.loc[i, 'mode']
            X_compose[mode, :] = X_base[sample, mode, :]
        return X_compose, predicted_features

    def sampling(self, features, N_realizations=100):
        df = az.summary(self.idata)

        # Get the mean value
        sigma = df.loc['eps', 'mean']
        a = df.loc['a', 'mean']
        weights_samples =[]
        for mode in range(self.config['N_modes']):
            mu = a
            for i, var_name in enumerate(self.var_names):
                b_i = df.loc[var_name, 'mean']
                mu += b_i*self.standrdizers[var_name].transform(features[mode, i].reshape(-1, 1))

            w_preds = norm.rvs(loc=mu, scale=sigma, size=(1,N_realizations))
            weights_samples.append(w_preds)
        return weights_samples

    def counterfactual_plot(self, x, cvar_name, N_samples=100):
        assert len(x) == len(self.target)

        # set control variable
        self.inputs_shared[cvar_name].set_value(x) 

        print("Sample posterior...")
        with self.model:
            post_pred = pm.sample_posterior_predictive(self.idata.posterior)
        print("Done")

        # plot the hdi of the prediction scatter
        _, ax = plt.subplots()
        az.plot_hdi(x, post_pred['w'])
        ax.plot(x, post_pred['w'].mean(0))
        ax.scatter(self.inputs[cvar_name], self.target)
        plt.show()

    def validation(self, config):
        df = az.summary(self.idata)
        # Get the mean value
        a = df.loc['a', 'mean']
        w_preds = a
        for var_name in self.var_names:
            b_i = df.loc[var_name, 'mean']
            # w_preds += b_i*self.standrdizers[var_name].transform(self.inputs[var_name])
            w_preds += b_i*self.inputs[var_name]
        
        self.feature_table['w_pred_mean'] = w_preds


        # Iterate through the input wp
        samples = self.feature_table['sample'].unique()
        weights = {sample:{} for sample in samples}
        for i in range(len(self.feature_table)):
            sample = self.feature_table.loc[i, 'sample']
            mode = self.feature_table.loc[i, 'mode']
            w_pred = self.feature_table.loc[i, 'w_pred_mean']
            weights[sample][mode] = w_pred

        print(self.feature_table)
        print(weights[sample])
        
        return