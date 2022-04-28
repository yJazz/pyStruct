import numpy as np

class ReconstructorInterface:
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class LinearReconstruction(ReconstructorInterface):
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.matmul(weights, X).flatten()

class InterceptReconstruction(ReconstructorInterface):
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        N_mode, _ = X.shape
        return weights[-1] + np.matmul(weights[:N_mode], X).flatten()