""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE

        # 1. Motion update (prediction)
        for i in range(self.num_particles):
            self.particles[i, :] = env.forward(self.particles[i], env.sample_noisy_action(u, self.alphas)).ravel()

        # 2. Compute weights based on observation likelihood
        for i in range(self.num_particles):
            z_hat = env.observe(self.particles[i], marker_id)
            innovation = np.array([[minimized_angle(z[0, 0] - z_hat[0, 0])]])
            self.weights[i] = env.likelihood(innovation, self.beta)

        # 3. Normalize weights
        self.weights += 1e-300  # avoid division by zero
        self.weights /= np.sum(self.weights)

        # 4. Resample
        self.particles, self.weights = self.resample(self.particles, self.weights)

        # 5. Estimate mean and covariance
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE

        n = self.num_particles
        new_particles = np.zeros_like(particles)

        r = np.random.uniform(0, 1 / n)
        c = weights[0]
        i = 0

        for m in range(n):
            u = r + m / n
            while u > c:
                i += 1
                c += weights[i]
            new_particles[m] = particles[i]

        new_weights = np.ones(n) / n
        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
