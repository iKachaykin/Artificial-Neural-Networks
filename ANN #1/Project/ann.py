import numpy as np
import tqdm as tqdm


class PerceptronSmplst:

    def __init__(self, param_n, step=0.0, epoch_n=10, eta=1.0):
        self.step = step
        self.param_n = param_n
        self.weights = np.zeros(param_n + 1)
        self.error = 0.0
        self.errors = []
        self.imax = epoch_n
        self.eta = eta

    def fit(self, sample, output):
        sample_expd = np.vstack((np.ones(sample.shape[0]), sample.T)).T
        for _ in tqdm.tqdm(range(self.imax)):
            y = np.where(np.dot(sample_expd, self.weights) > self.step, 1.0, -1.0)
            self.weights -= self.eta * np.dot((y - output), sample_expd)

    def predict(self, vect):
        return np.dot(self.weights[1:], vect) + self.weights[0]

    def classify(self, vect):
        if self.predict(vect) > self.step:
            return 1
        else:
            return -1


class PerceptronLgst:

    def __init__(self, param_n, layer_n, neuron_n, epoch_n=10, activation_param=1.0, learning_rate=1.0, momentum_c=0.0,
                 w_lower_bound=-0.9, w_upper_bound=0.9):
        if layer_n < 1:
            raise ValueError('Number of layers must be positive non-zero number!')
        if neuron_n.size != layer_n:
            raise ValueError('neuron_n.size must be equal to layer_n!')
        self.param_n = param_n
        self.layer_n = layer_n
        self.neuron_n = neuron_n.copy()
        self.epoch_n = epoch_n
        self.activation_param = activation_param
        self.learning_rate = learning_rate
        self.momentum_c = momentum_c
        self.weights = []
        self.errors = []
        self.test_errors = []
        self.error = 0.0
        for i in range(layer_n):
            tmp = []
            for j in range(neuron_n[i]):
                if i == 0:
                    # tmp.append(np.ones(self.param_n + 1))
                    tmp.append(np.random.rand(self.param_n + 1) * (w_upper_bound - w_lower_bound) + w_lower_bound)
                else:
                    # tmp.append(np.ones(self.neuron_n[i-1] + 1))
                    tmp.append(np.random.rand(self.neuron_n[i-1] + 1) * (w_upper_bound - w_lower_bound) + w_lower_bound)
            tmp = np.array(tmp)
            self.weights.append(tmp)

    def activation(self, x):
        return 1 / (1 + np.exp(-self.activation_param * x))

    def fit_once(self, x, desired_output):
        y = [np.concatenate(([1], x.copy()))]
        for i in range(self.layer_n):
            y.append(self.activation(np.dot(self.weights[i], y[i])))
            if i != self.layer_n - 1:
                y[i+1] = np.concatenate(([1], y[i+1]))
        self.error += np.sum((desired_output - y[self.layer_n]) ** 2) / 2
        local_grads = []
        for i in range(self.layer_n, 0, -1):
            if i == self.layer_n:
                local_grads.append(self.activation_param * (y[i] - desired_output) *
                                        y[i] * (1 - y[i]))
            else:
                local_grads.append(self.activation_param * y[i][1:] * (1 - y[i][1:]) *
                                   np.dot(local_grads[-1], self.weights[i][:, 1:]))
        local_grads = local_grads[::-1]
        for i in range(self.layer_n):
            self.weights[i] = self.weights[i] - self.learning_rate * \
                              local_grads[i].reshape(local_grads[i].size, 1) * \
                              (y[i] * np.ones((local_grads[i].size, y[i].size)))

    def fit(self, sample, output, test_sample, test_output):
        self.errors = []
        self.test_errors = []
        for _ in tqdm.tqdm(range(self.epoch_n)):
            order = np.arange(sample.shape[0])
            np.random.shuffle(order)
            for i in order:
                self.fit_once(sample[i], output[i])
            self.errors.append(self.error / sample.shape[0])
            self.error = 0.0
            for i in range(test_sample.shape[0]):
                self.error += np.sum((test_output[i] - self.predict(test_sample[i])) ** 2) / 2
            self.test_errors.append(self.error / test_sample.shape[0])
            self.error = 0.0
        self.errors = np.array(self.errors)
        self.test_errors = np.array(self.test_errors)

    def predict(self, x):
        y = [np.concatenate(([1], x.copy()))]
        for i in range(self.layer_n):
            y.append(self.activation(np.dot(self.weights[i], y[i])))
            if i != self.layer_n - 1:
                y[i + 1] = np.concatenate(([1], y[i + 1]))
        return y[-1]
