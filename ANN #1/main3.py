import numpy as np
import ann as ann
import matplotlib.pyplot as plt


def f(t):
    return (1.0 + 2 * np.cos(t)) / 2.0


if __name__ == '__main__':

    figsize = (15.0, 7.5)
    plt.figure(figsize=figsize)

    t_left, t_right, grid_dot_num = -np.pi, np.pi, 2000
    w_lower_bound, w_upper_bound = -1.0, 1.0
    t = np.linspace(t_left, t_right, grid_dot_num)
    y = f(t)
    y_min, y_max = y.min(), y.max()
    param_n, layer_n, neuron_n, epoch_n, activation_param, learning_rate =\
        1, 2, np.array([200, 1]), 1000, 2.0, 0.05
    pl = ann.PerceptronLgst(param_n, layer_n, neuron_n, epoch_n, activation_param, learning_rate,
                            w_lower_bound=w_lower_bound, w_upper_bound=w_upper_bound)
    sample, output = t.reshape(t.size, -1), (y.reshape(y.size, -1) - y_min) / (y_max - y_min)

    # plt.plot(sample.ravel(), output.ravel(), 'b-')

    test_vol = 1000
    t_test = np.random.rand(test_vol) * (t_right - t_left) + t_left
    t_test = np.sort(t_test)
    y_test = f(t_test)
    y_test_min, y_test_max = y_test.min(), y_test.max()
    test_sample, test_output = \
        t_test.reshape(t_test.size, -1),\
        (y_test.reshape(y_test.size, -1) - y_test_min) / (y_test_max - y_test_min)

    pl.fit(sample, output)
    print(pl.errors[-1])

    y_predicted = []
    for ti in sample:
        y_predicted.append(pl.predict(ti)[0] * (y_max - y_min) + y_min)
    y_predicted = np.array(y_predicted)

    y_test_predicted = []
    for ti in test_sample:
        y_test_predicted.append(pl.predict(ti)[0] * (y_max - y_min) + y_min)
    y_test_predicted = np.array(y_test_predicted)
    print(np.linalg.norm(y_test_predicted - y_test))

    plt.grid(True)
    plt.plot(t_test, y_test, 'k-', t_test, y_test_predicted, 'r-')
    plt.show()
    plt.close()
