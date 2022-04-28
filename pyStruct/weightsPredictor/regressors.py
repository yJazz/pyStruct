from scipy.stats import percentileofscore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class WeightsPredictor:
    def train(self, sample_set):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
        




