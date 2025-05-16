import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import minimized_angle
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter
from utils import minimized_angle, plot_field, plot_robot, plot_path


def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = \
        env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        fig = env.get_figure()

    for i in range(num_steps):
        x_real = states_real[i + 1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i + 1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(env, states_noisefree[:i + 1, :], 'g', 0.5)
            plot_path(env, states_real[:i + 1, :], 'b')
            if filt is not None:
                plot_path(env, states_filter[:i + 1, :2], 'r')
            fig.canvas.flush_events()

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
            errors[i:i + 1, :].dot(np.linalg.inv(cov)).dot(errors[i:i + 1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)

    if plot:
        plt.show(block=True)

    return position_errors, mahalanobis_errors


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--plot', action='store_true',
        help='turn on plotting')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')

    return parser


def analyze_pf_performance():
    # Default parameters
    alphas = np.array([0.05 ** 2, 0.005 ** 2, 0.1 ** 2, 0.01 ** 2])
    beta = np.diag([np.deg2rad(5) ** 2])

    # Noise factors to analyze
    noise_factors = [1 / 64, 1 / 16, 1 / 4, 4, 16, 64]
    particle_counts = [20, 50, 500]

    # Analysis for part b)
    print("Running analysis for part b)...")
    mean_position_errors_b = []
    for r in noise_factors:
        data_factor = r
        filter_factor = r

        env = Field(data_factor * alphas, data_factor * beta)
        policy = policies.OpenLoopRectanglePolicy()

        initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
        initial_cov = np.diag([10, 10, 1])

        filt = ParticleFilter(
            initial_mean,
            initial_cov,
            100,  # Default number of particles
            filter_factor * alphas,
            filter_factor * beta
        )

        # Run 10 trials
        trials_errors = []
        for _ in range(10):
            errors, _ = localize(env, policy, filt, initial_mean, 5, False)
            trials_errors.append(np.mean(errors))

        mean_position_errors_b.append(np.mean(trials_errors))

    # Analysis for part c)
    print("Running analysis for part c)...")
    mean_position_errors_c = []
    anees_values_c = []
    for r in noise_factors:
        data_factor = 1  # Default data factor
        filter_factor = r

        env = Field(data_factor * alphas, data_factor * beta)
        policy = policies.OpenLoopRectanglePolicy()

        initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
        initial_cov = np.diag([10, 10, 1])

        filt = ParticleFilter(
            initial_mean,
            initial_cov,
            100,  # Default number of particles
            filter_factor * alphas,
            filter_factor * beta
        )

        # Run 10 trials
        trials_position_errors = []
        trials_anees = []
        for _ in range(10):
            position_errors, mahalanobis_errors = localize(env, policy, filt, initial_mean, 5, False)
            mean_position_error = np.mean(position_errors)
            anees = np.mean(mahalanobis_errors) / 3

            trials_position_errors.append(mean_position_error)
            trials_anees.append(anees)

        mean_position_errors_c.append(np.mean(trials_position_errors))
        anees_values_c.append(np.mean(trials_anees))

    # Analysis for part d)
    print("Running analysis for part d)...")
    mean_position_errors_d = []
    anees_values_d = []
    for pc in particle_counts:
        mean_errors_pc = []
        anees_pc = []
        for r in noise_factors:
            data_factor = r
            filter_factor = r

            env = Field(data_factor * alphas, data_factor * beta)
            policy = policies.OpenLoopRectanglePolicy()

            initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
            initial_cov = np.diag([10, 10, 1])

            filt = ParticleFilter(
                initial_mean,
                initial_cov,
                pc,  # Varying number of particles
                filter_factor * alphas,
                filter_factor * beta
            )

            # Run 10 trials
            trials_position_errors = []
            trials_anees = []
            for _ in range(10):
                position_errors, mahalanobis_errors = localize(env, policy, filt, initial_mean, 5, False)
                mean_position_error = np.mean(position_errors)
                anees = np.mean(mahalanobis_errors) / 3

                trials_position_errors.append(mean_position_error)
                trials_anees.append(anees)

            mean_errors_pc.append(np.mean(trials_position_errors))
            anees_pc.append(np.mean(trials_anees))

        mean_position_errors_d.append(mean_errors_pc)
        anees_values_d.append(anees_pc)

    # Plot results
    plt.figure(figsize=(9, 9))

    # Part b) plot
    plt.subplot(1, 2, 1)
    plt.plot(noise_factors, mean_position_errors_b, marker='o')
    plt.xscale('log')
    plt.xlabel('Noise Factor (r)')
    plt.ylabel('Mean Position Error')
    plt.title('Mean Position Error vs Noise Factor (Part b)')
    plt.grid(True)

    # Part c) plot
    plt.subplot(1, 2, 2)
    # plt.plot(noise_factors, mean_position_errors_c, marker='o', label='Mean Position Error')
    plt.plot(noise_factors, anees_values_c, marker='x', label='ANEES')
    plt.xscale('log')
    plt.xlabel('Noise Factor (r)')
    plt.ylabel('Error Values')
    plt.title('ANEES vs Noise Factor (Part c)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(9, 9))


    # Part d) plot - Mean Position Error
    plt.subplot(1, 2, 1)
    for i, pc in enumerate(particle_counts):
        plt.plot(noise_factors, mean_position_errors_d[i], marker='o', label=f'{pc} particles')
    plt.xscale('log')
    plt.xlabel('Noise Factor (r)')
    plt.ylabel('Mean Position Error')
    plt.title('Mean Position Error vs Noise Factor for Different Particle Counts (Part d)')
    plt.legend()
    plt.grid(True)

    # Part d) plot - ANEES
    plt.subplot(1, 2, 2)
    for i, pc in enumerate(particle_counts):
        plt.plot(noise_factors, anees_values_d[i], marker='x', label=f'{pc} particles')
    plt.xscale('log')
    plt.xlabel('Noise Factor (r)')
    plt.ylabel('ANEES')
    plt.title('ANEES vs Noise Factor for Different Particle Counts (Part d)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    args = setup_parser().parse_args()

    if args.filter_type == 'pf':
        # Run the localization with PF
        alphas = np.array([0.05 ** 2, 0.005 ** 2, 0.1 ** 2, 0.01 ** 2])
        beta = np.diag([np.deg2rad(5) ** 2])

        env = Field(args.data_factor * alphas, args.data_factor * beta)
        policy = policies.OpenLoopRectanglePolicy()

        initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
        initial_cov = np.diag([10, 10, 1])

        filt = ParticleFilter(
            initial_mean,
            initial_cov,
            args.num_particles,
            args.filter_factor * alphas,
            args.filter_factor * beta
        )

        localize(env, policy, filt, initial_mean, args.num_steps, args.plot)

        # Run performance analysis
        analyze_pf_performance()
    elif args.filter_type == 'ekf':
        # EKF implementation (from previous answer)
        pass
    else:
        # Handle other filter types if needed
        pass