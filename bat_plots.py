import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

np.random.seed(42)

response_str = 'Performance'
true_response_str = "True " + response_str
true_x_str = 'Tutoring hours'


def black_box_function(x):
    y = 1 / (1 + np.exp(-(x - 6))) + 1
    return y


def experiment1(lambda_u1=300, lambda_u2=100, num_total_samples=12, noise_std=0.05):
    # range of x values
    x_range = np.linspace(0, 4 * np.pi, 100)

    true_response = black_box_function(x_range)

    true_response_deriv = np.array(
        [0 if i == 0 else true_response[i] - true_response[i - 1] for i in range(0, len(true_response))])

    exploitive_u = true_response + lambda_u1 * true_response_deriv

    x_gold = x_range[exploitive_u.argmax()]

    # ZOOM PLOT:
    zoom_x_lims = [x_gold - 1, x_gold + 1]

    # FIXED SAMPLES:
    plot_fixed_samples = True
    if plot_fixed_samples:
        sample_x = np.linspace(0, 4 * np.pi, num_total_samples)

        # output for each sampled x value
        sample_y = black_box_function(sample_x) + np.random.normal(loc=0.0, scale=noise_std, size=sample_x.shape)

        # Gaussian process regressor with an RBF kernel
        kernel = RBF(length_scale=2)
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=(noise_std) ** 2)

        # Fit the Gaussian process model to the data points
        gp_model.fit(sample_x.reshape(-1, 1), sample_y)

        print(gp_model.get_params())

        # Generate predictions using the GP model
        y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)

        # Plot
        plt.plot(x_range, black_box_function(x_range), label=true_response_str, color='red', linestyle='dashed',
                 alpha=0.6)
        plt.scatter(sample_x, sample_y, color='red', label='Samples')
        plt.plot(x_range, y_pred, color='green', label='GP ($\mu \pm 2\sigma$)')
        print('y_std.max: ', y_std.max())
        plt.fill_between(x_range, y_pred - 2 * y_std, y_pred + 2 * y_std, color='green', alpha=0.2)
        plt.xlabel(true_x_str)
        plt.ylabel(response_str)
        plt.ylim(0.8, 2.2)

        plt.axvline(x=x_gold, color='y', linestyle='dashed')
        plt.title('{r} with Gaussian Process Surrogate Model'.format(r=response_str))
        plt.legend()
        plt.show()

        plt.plot(x_range, black_box_function(x_range), label=true_response_str, color='red', linestyle='dashed',
                 alpha=0.6)
        plt.scatter(sample_x, sample_y, color='red', label='Samples')
        plt.plot(x_range, y_pred, color='green', label='GP ($\mu \pm 2\sigma$)')
        print('y_std.max: ', y_std.max())

        plt.fill_between(x_range, y_pred - 2 * y_std, y_pred + 2 * y_std, color='green', alpha=0.2)
        plt.xlabel(true_x_str)
        plt.ylabel(response_str)
        plt.ylim(1.2, 1.8)
        plt.xlim(zoom_x_lims[0], zoom_x_lims[1])
        plt.axvline(x=x_gold, color='y', linestyle='dashed')
        plt.legend()
        plt.show()

    plot_BAT_samples = True
    if plot_BAT_samples:
        y_min, y_max = 0.0, 2.5
        samples_x = np.array([8.0])
        samples_y = black_box_function(samples_x) + np.random.normal(loc=0.0, scale=noise_std, size=samples_x.shape)

        for num_adapt_samples in range(0, num_total_samples):
            # Gaussian process regressor with an RBF kernel
            kernel = RBF(length_scale=2)
            gp_model = GaussianProcessRegressor(kernel=kernel, alpha=(noise_std) ** 2)

            # Fit the Gaussian process model to the sampled points
            gp_model.fit(samples_x.reshape(-1, 1), samples_y - samples_y.mean())

            # Generate predictions using the GP model
            y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)
            plt.plot(x_range, black_box_function(x_range), label=true_response_str, color='red', linestyle='dashed',
                     alpha=0.6)
            plt.scatter(samples_x, samples_y, color='red', label='Samples')
            plt.plot(x_range, y_pred + samples_y.mean(), color='blue', label='GP ($\mu \pm 2\sigma$)')
            plt.fill_between(x_range, y_pred + samples_y.mean() - 2 * y_std, y_pred + samples_y.mean() + 2 * y_std,
                             color='blue', alpha=0.2)
            plt.xlabel(true_x_str)
            plt.ylim(y_min, y_max)

            y_pred_deriv = np.array([0 if i == 0 else y_pred[i] - y_pred[i - 1] for i in range(0, len(y_pred))])
            u1_deriv = y_pred_deriv * lambda_u1
            plt.plot(x_range[1:], u1_deriv[1:] / u1_deriv.max(), color='black', alpha=0.7,
                     label='$u_1 \propto \partial \mu/ \partial h$',
                     linestyle='dashed')

            u2_sigma = lambda_u2 * y_std
            plt.plot(x_range, u2_sigma / u2_sigma.max(), color='black', alpha=0.7, label='$u_2 \propto \sigma$',
                     linestyle='dotted')

            u_total = y_pred + u1_deriv + u2_sigma
            u_total = u_total - u_total.min()
            plt.plot(x_range[1:], u_total[1:] / u_total[1:].max(), color='black', alpha=0.7,
                     label='$u_3 = \mu + \lambda_1 u_1 + \lambda_2 u_2$',
                     linestyle='solid', linewidth=2)

            next_index = u_total[1:-1].argmax()
            next_sample_x = x_range[next_index]
            next_sample_y = black_box_function(next_sample_x) \
                            + np.random.normal(loc=0.0, scale=noise_std, size=1)
            samples_x = np.append(samples_x, next_sample_x)
            samples_y = np.append(samples_y, next_sample_y)

            plt.axvline(x=x_gold, color='y', linestyle='dashed')  # , label='axvline')
            plt.legend()
            plt.show()

        # Final plot:
        plt.plot(x_range, black_box_function(x_range), label=true_response_str, color='red', linestyle='dashed',
                 alpha=0.6)
        plt.scatter(samples_x, samples_y, color='red', label='Samples')
        plt.plot(x_range, y_pred + samples_y.mean(), color='blue', label='GP ($\mu \pm 2\sigma$)')
        plt.fill_between(x_range, y_pred + samples_y.mean() - 2 * y_std, y_pred + samples_y.mean() + 2 * y_std,
                         color='blue', alpha=0.2)
        plt.xlabel(true_x_str)
        plt.axvline(x=x_gold, color='y', linestyle='dashed')  # , label='axvline')
        plt.ylim(1.2, 1.8)
        plt.xlim(zoom_x_lims[0], zoom_x_lims[1])
        plt.legend()
        plt.show()
    sys.exit()


def main():
    plt.interactive(False)
    experiment1()


if __name__ == "__main__":
    main()
    exit()
