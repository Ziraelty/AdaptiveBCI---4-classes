

import numpy as np
from scipy.linalg import expm


def _powm(A, p):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(vals ** p) @ vecs.T


def _logm(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(np.log(vals)) @ vecs.T

def _expm(S):
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T


def _sym(A):
    return (A + A.T) / 2



def covariance(X):

    M, T = X.shape
    assert T > M
    X = X - X.mean(axis=1, keepdims=True)
    C = (X @ X.T) / (T - 1)
    return _sym(C)



def dist_riemann(A, B):
    A_neg_half = _powm(A, -0.5)
    inner = _sym(A_neg_half @ B @ A_neg_half)
    vals = np.linalg.eigvalsh(inner)
    vals = np.maximum(vals, 1e-12)
    return np.sqrt(np.sum(np.log(vals) ** 2))


def riemannian_mean(covs, n_iter=50, tol=1e-7):
    covs = np.array(covs)
    M = np.mean(covs, axis=0)

    for _ in range(n_iter):
        M_half = _powm(M, 0.5)
        M_neg_half = _powm(M, -0.5)

        S = np.zeros_like(M)
        for C in covs:
            inner = _sym(M_neg_half @ C @ M_neg_half)
            S += _logm(inner)
        S /= len(covs)

        if np.linalg.norm(S, 'fro') < tol:
            break

        M = _sym(M_half @ _expm(S) @ M_half)

    return M



def estimate_R(rest_covs):
    return riemannian_mean(rest_covs)


def rpa_recenter(covs, R):
    R_neg_half = _powm(R, -0.5)
    covs = np.array(covs)
    single = covs.ndim == 2
    if single:
        covs = covs[np.newaxis]
    aligned = np.array([
        _sym(R_neg_half @ C @ R_neg_half)
        for C in covs
    ])
    return aligned[0] if single else aligned



def riemannian_ema(M, C_new, alpha=0.01):
    M_half = _powm(M, 0.5)
    M_neg_half = _powm(M, -0.5)
    inner = _sym(M_neg_half @ C_new @ M_neg_half)
    log_inner = _logm(inner)
    return _sym(M_half @ _expm(alpha * log_inner) @ M_half)



class MDM:


    def __init__(self, n_iter=50, tol=1e-7):
        self.n_iter = n_iter
        self.tol = tol
        self.centroids_ = {}
        self.sigmas_ = {}
        self.classes_ = None

    def fit(self, covs, labels):
        covs = np.array(covs)
        self.classes_ = np.unique(labels)

        for c in self.classes_:
            idx = np.where(labels == c)[0]
            class_covs = covs[idx]

            self.centroids_[c] = riemannian_mean(
                class_covs, n_iter=self.n_iter, tol=self.tol
            )

            dists = np.array([
                dist_riemann(C, self.centroids_[c])
                for C in class_covs
            ])
            self.sigmas_[c] = np.std(dists) + 1e-10

        return self

    def predict(self, covs):
        covs = np.array(covs)
        single = covs.ndim == 2
        if single:
            covs = covs[np.newaxis]

        preds = []
        for C in covs:
            dists = {c: dist_riemann(C, self.centroids_[c]) for c in self.classes_}
            preds.append(min(dists, key=dists.get))

        result = np.array(preds)
        return result[0] if single else result

    def predict_bayes(self, covs):
        covs = np.array(covs)
        single = covs.ndim == 2
        if single:
            covs = covs[np.newaxis]

        n = list(self.centroids_.values())[0].shape[0]
        # log ζ(σ) ≈ n(n+1)/2 · log(σ) dla rozkładu Gaussa na P(n)
        # Źródło: Zanini 2018 eq.8, przybliżenie dla dużych σ
        dim_factor = n * (n + 1) / 2

        preds = []
        for C in covs:
            scores = {}
            for c in self.classes_:
                d2 = dist_riemann(C, self.centroids_[c]) ** 2
                sigma = self.sigmas_[c]
                scores[c] = dim_factor * np.log(sigma) + d2 / (2 * sigma ** 2)
            preds.append(min(scores, key=scores.get))

        result = np.array(preds)
        return result[0] if single else result

    def update_centroid(self, label, C_new, alpha=0.01):
        if label not in self.centroids_:
            raise ValueError(f"Nieznana klasa: {label}")
        self.centroids_[label] = riemannian_ema(
            self.centroids_[label], C_new, alpha=alpha
        )

    def score(self, covs, labels, bayes=False):
        covs = np.array(covs)
        preds = self.predict_bayes(covs) if bayes else self.predict(covs)
        return np.mean(preds == labels)



