import numpy as np

class ReconstructorInterface:
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class LinearReconstruction(ReconstructorInterface):
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.matmul(weights, X).flatten()

class InterceptReconstruction(ReconstructorInterface):
    def reconstruct(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return weights[0] + np.matmul(weights[1:], X).flatten()