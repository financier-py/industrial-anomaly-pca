import numpy as np

class PCAnumpy:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> "PCAnumpy":
        '''Находим главные оси по тренировочным данным'''
        # центрируем
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[:self.n_components]

        # считаем дисперсию
        explained_variance = S ** 2 / (len(X) - 1)
        self.explained_variance_ = explained_variance[:self.n_components]
        self.explained_variance_ratio_ = (
            explained_variance[:self.n_components] / explained_variance.sum()
        )

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        '''Проецируем данные на главные оси'''
        X_centered = X - self.mean_
        return X_centered @ self.components_.T # pyright: ignore[reportOptionalMemberAccess]
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        '''Обратная проекция'''
        return X_transformed @ self.components_ + self.mean_
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.inverse_transform(self.transform(X))
