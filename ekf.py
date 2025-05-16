""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # Prediction step
        state_input = self.mu
        control_input = u

        G = env.G(state_input, control_input)
        V = env.V(state_input, control_input)
        R = env.noise_from_motion(control_input, self.alphas)

        mu_bar = env.forward(state_input.ravel(), control_input.ravel())  # Predicted mean
        sigma_bar = G @ self.sigma @ G.T + V @ R @ V.T  # Predicted covariance

        # Correction step
        landmark_pos = np.array([
            env.MARKER_X_POS[marker_id],
            env.MARKER_Y_POS[marker_id]
        ]).reshape((-1, 1))

        z_hat = env.observe(mu_bar.ravel(), marker_id)  # Expected observation
        H = env.H(mu_bar, marker_id)  # Jacobian of observation model
        Q = self.beta  # Observation noise

        innovation = minimized_angle(z[0, 0] - z_hat[0, 0])
        S = H @ sigma_bar @ H.T + Q  # Innovation covariance
        K = sigma_bar @ H.T @ np.linalg.inv(S)  # Kalman Gain

        mu_bar += K @ np.array([[innovation]])
        mu_bar[2] = minimized_angle(mu_bar[2])  # Normalize theta

        sigma_bar = (np.eye(3) - K @ H) @ sigma_bar

        # Save and return state
        self.mu = mu_bar
        self.sigma = sigma_bar
        return self.mu, self.sigma
