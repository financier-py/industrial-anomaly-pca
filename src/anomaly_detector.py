import numpy as np
from sklearn.preprocessing import StandardScaler
from pca_numpy import PCAnumpy

class AnomalyDetector:
    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self.pca = PCAnumpy(n_components)
        self.scaler = StandardScaler()
        self.threshold = None
    
    def fit(self, X_normal: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_normal)
        self.pca.fit(X_scaled)
        errors = self._reconstruction_error(X_scaled)

        self.threshold = errors.mean() + 3 * errors.std()

        return self
    
    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        X_reconstructed = self.pca.reconstruct(X_scaled)
        return np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self._reconstruction_error(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        return (scores > self.threshold).astype(int)