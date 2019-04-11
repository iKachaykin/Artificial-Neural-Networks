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

    def fit(self, sample, output, test_sample=None, test_output=None):
        self.errors = []
        self.test_errors = []
        for _ in tqdm.tqdm(range(self.epoch_n)):
            order = np.arange(sample.shape[0])
            np.random.shuffle(order)
            for i in order:
                self.fit_once(sample[i], output[i])
            self.errors.append(self.error / sample.shape[0])
            self.error = 0.0
            if test_sample is not None and test_output is not None:
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


class HopfieldNetworkTSP:

    def __init__(self, cnum, mu=(950, 2500, 1500, 475, 2500), A=0.0005, B=0.008, C=0.008, activation_param=20.0,
                 eps=10e-5, iter_lim=1000):

        self.cnum = cnum
        self.mu = mu
        self.A = A
        self.B = B
        self.C = C
        self.activation_param = activation_param
        self.eps = eps
        self.iter_lim = iter_lim

        self.weights = np.zeros((cnum, cnum, cnum, cnum))
        self.bias = np.zeros((cnum, cnum))
        self.net_input_current = np.zeros((cnum, cnum))
        self.net_input_prev = np.zeros((cnum, cnum))
        self.net_output = np.zeros((cnum, cnum))

        self.distances = None
        self.existances = None
        self.start = None
        self.destination = None

        for x in range(self.weights.shape[0]):
            for i in range(self.weights.shape[1]):
                for y in range(self.weights.shape[2]):
                    for j in range(self.weights.shape[3]):
                        self.weights[x, i, y, j] = self.mu[3] * self.kdelta(x, y) * self.kdelta(i, j) - \
                                                   self.mu[2] * (self.kdelta(x, y) + self.kdelta(i, j) -
                                                                 self.kdelta(j, x) - self.kdelta(i, y))

    def set_problem(self, distances, existances, start, destination):
        self.distances = np.array(distances)
        self.existances = np.array(existances)
        self.start = start
        self.destination = destination

        for x in range(self.cnum):
            for i in range(self.cnum):
                self.bias[x, i] = -self.mu[0] / 2 * self.distances[x, i] * \
                                  (1 - self.kdelta(x, self.destination) * self.kdelta(i, self.start)) - \
                                  self.mu[1] / 2 * self.existances[x, i] * \
                                  (1 - self.kdelta(x, self.destination) * self.kdelta(i, self.start)) - \
                                  self.mu[3] / 2 + self.mu[4] / 2 * \
                                  self.kdelta(x, self.destination) * self.kdelta(i, self.start)

        self.net_input_current = self.distances.copy()
        self.net_output = self.activation(self.net_input_current)

    def solve(self):
        if self.distances is None or self.existances is None or self.start is None or self.destination is None:
            return None
        count = 0
        while np.linalg.norm(self.net_input_current - self.net_input_prev) > self.eps and count < self.iter_lim:
            self.net_input_current, self.net_input_prev = \
                (self.net_input_current - self.A * self.net_input_prev +
                 self.B * self.weights_output_sub_dot() + self.C * self.bias,
                 self.net_input_current)
            self.net_output = self.activation(self.net_input_current)
            count += 1
        return self.net_output.copy()

    def activation(self, x):
        # return (1.0 + np.tanh(x / self.activation_param)) / 2.0
        return 1 / (1 + np.exp(-self.activation_param * x))

    def kdelta(self, x, y):
        return 1 if x == y else 0

    def weights_output_sub_dot(self):
        result = np.zeros((self.cnum, self.cnum))
        for x in range(self.weights.shape[0]):
            for i in range(self.weights.shape[1]):
                for y in range(self.weights.shape[2]):
                    for j in range(self.weights.shape[3]):
                        result[x, i] += self.weights[x, i, y, j] * self.net_output[y, j] if y != j else 0
        return result


class HopfieldNetworkTSPAlt:

    def __init__(self, cnum, A=500, B=500, C=200, D=500, activation_param=0.02,
                 eps=10e-5, iter_lim=1000):

        self.cnum = cnum
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.activation_param = activation_param
        self.eps = eps
        self.iter_lim = iter_lim

        self.weights = np.zeros((cnum, cnum, cnum, cnum))
        self.bias = np.zeros((cnum, cnum))
        self.net_input_current = np.zeros((cnum, cnum))
        self.net_input_prev = np.zeros((cnum, cnum))
        self.net_output = np.zeros((cnum, cnum))

        self.distances = None

    def set_problem(self, distances):
        self.distances = np.array(distances)

        for x in range(self.weights.shape[0]):
            for i in range(self.weights.shape[1]):
                for y in range(self.weights.shape[2]):
                    for j in range(self.weights.shape[3]):
                        self.weights[x, i, y, j] = -self.A * self.kdelta(x, y) * (1 - self.kdelta(i, j)) + \
                                                   -self.B * self.kdelta(i, j) * (1 - self.kdelta(x, y)) + \
                                                   -self.C - self.D * self.distances[x, y] * \
                                                   (self.kdelta(j, i+1) + self.kdelta(j, i-1))

        self.net_input_current = self.distances.copy()
        self.net_output = self.activation(self.net_input_current)

    def solve(self):
        if self.distances is None:
            return None
        count = 0
        while np.linalg.norm(self.net_input_current - self.net_input_prev) > self.eps and count < self.iter_lim:
            tmp = self.net_input_current.copy()
            for x in range(self.cnum):
                for i in range(self.cnum):
                    self.net_input_current[x, i] = tmp[x, i] - self.A * (self.net_output[x, 0:i].sum() +
                                                                         self.net_output[x, i+1:self.cnum].sum()) + \
                                                   -self.B * (self.net_output[0:x, i].sum() +
                                                              self.net_output[x+1:self.cnum, i].sum()) + \
                                                   -self.C * (self.net_output.sum() - self.cnum) + \
                                                   -self.D * (self.distances[x] *
                                                              (self.net_output[:, i+1 if i+1 < self.cnum else 0] *
                                                               float(1 if i+1 < self.cnum else 0) +
                                                               self.net_output[:, i-1 if i-1 >= 0 else 0] *
                                                               float(1 if i-1 >= 0 else 0))).sum()
            self.net_input_prev = tmp.copy()
            self.net_output = self.activation(self.net_input_current)
            count += 1
        return self.net_output.copy()

    def activation(self, x):
        return (1.0 + np.tanh(x / self.activation_param)) / 2.0

    def kdelta(self, x, y):
        return 1 if x == y else 0
