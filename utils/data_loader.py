import numpy as np

def load_eeg_or_simulate(n_subjects=200, n_features=16, seed=0):
    rng = np.random.RandomState(seed)
    try:
        raise RuntimeError('skip download in starter')
    except Exception:
        X_class0 = rng.normal(loc=0.0, scale=1.0, size=(n_subjects//2, n_features))
        X_class1 = rng.normal(loc=0.3, scale=1.0, size=(n_subjects - n_subjects//2, n_features))
        X = np.vstack([X_class0, X_class1])
        y = np.array([0]*(n_subjects//2) + [1]*(n_subjects - n_subjects//2))
        idx = rng.permutation(n_subjects)
        return X[idx], y[idx]
