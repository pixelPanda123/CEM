import numpy as np


class GaussianHMM:
    def __init__(self, n_states=2, eps=1e-6):
        self.n_states = n_states
        self.eps = eps

    # -----------------------------
    # Gaussian Emission
    # -----------------------------
    def _gaussian_pdf(self, x, mean, cov):
        d = len(mean)
        cov = cov + self.eps * np.eye(d)

        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        diff = x - mean
        exponent = -0.5 * diff.T @ cov_inv @ diff
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_cov)

        return norm_const * np.exp(exponent)

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize(self, X):
        T, d = X.shape

        self.pi = np.ones(self.n_states) / self.n_states

        self.A = np.array([[0.9, 0.1],
                           [0.1, 0.9]])

        indices = np.random.choice(T, self.n_states, replace=False)
        self.means = X[indices]

        self.covs = np.array([np.eye(d) for _ in range(self.n_states)])

    # -----------------------------
    # Forward Algorithm
    # -----------------------------
    def _forward(self, X):
        T = len(X)
        alpha = np.zeros((T, self.n_states))

        # Initial step
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self._gaussian_pdf(
                X[0], self.means[i], self.covs[i]
            )

        alpha[0] /= alpha[0].sum()

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                emission = self._gaussian_pdf(
                    X[t], self.means[j], self.covs[j]
                )

                alpha[t, j] = emission * np.sum(
                    alpha[t - 1] * self.A[:, j]
                )

            alpha[t] /= alpha[t].sum()

        return alpha

    # -----------------------------
    # Backward Algorithm
    # -----------------------------
    def _backward(self, X):
        T = len(X)
        beta = np.zeros((T, self.n_states))

        beta[-1] = 1.0

        for t in reversed(range(T - 1)):
            for i in range(self.n_states):
                total = 0
                for j in range(self.n_states):
                    emission = self._gaussian_pdf(
                        X[t + 1], self.means[j], self.covs[j]
                    )
                    total += (
                        self.A[i, j]
                        * emission
                        * beta[t + 1, j]
                    )

                beta[t, i] = total

            beta[t] /= beta[t].sum()

        return beta

    # -----------------------------
    # Compute Gamma (Posterior)
    # -----------------------------
    def _compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    # -----------------------------
    # Compute Xi
    # -----------------------------
    def _compute_xi(self, X, alpha, beta):
        T = len(X)
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        for t in range(T - 1):
            denom = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission = self._gaussian_pdf(
                        X[t + 1], self.means[j], self.covs[j]
                    )
                    xi[t, i, j] = (
                        alpha[t, i]
                        * self.A[i, j]
                        * emission
                        * beta[t + 1, j]
                    )
                    denom += xi[t, i, j]

            xi[t] /= denom

        return xi

    # -----------------------------
    # M-Step
    # -----------------------------
    def _m_step(self, X, gamma, xi):
        T, d = X.shape

        # Update initial probabilities
        self.pi = gamma[0]

        # Update transition matrix
        self.A = xi.sum(axis=0)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Update means
        for i in range(self.n_states):
            weight = gamma[:, i].sum()
            self.means[i] = (gamma[:, i][:, None] * X).sum(axis=0) / weight

        # Update covariances
        for i in range(self.n_states):
            cov = np.zeros((d, d))
            for t in range(T):
                diff = X[t] - self.means[i]
                cov += gamma[t, i] * np.outer(diff, diff)

            cov /= gamma[:, i].sum()
            self.covs[i] = cov + self.eps * np.eye(d)

    # -----------------------------
    # Training Loop
    # -----------------------------
    def fit(self, X, n_iter=20):
        self.initialize(X)

        for _ in range(n_iter):
            alpha = self._forward(X)
            beta = self._backward(X)
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(X, alpha, beta)

            self._m_step(X, gamma, xi)

        return gamma