import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize list of weight matrices and store
        # learning rate and network architecture
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.loss_data = []
        self.epoch_data = []

        # initialize weights
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append((w / np.sqrt(layers[i])))

        # initialize weights for 2nd to last layer
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return string that represents network architecture
        NN_design = "-".join(str(num_nodes) for num_nodes in self.layers)
        return f"NeuralNetwork: {NN_design}"

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1 - x)

    def fit(self, X, y, epochs=100, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        loss_data = []
        epoch_data = []

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f"[INFO] epoch={epoch+1}, loss={loss:.7f}")
                self.loss_data.append(loss)
                self.epoch_data.append(epoch)

    def fit_partial(self, x, y):
        # construct list of output activations for each layer
        # first is just input feature vector
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        for layer in np.arange(0, len(self.W)):
            # feedforward at current layer
            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            A.append(out)

        # BACKPROPAGATION:
        error = A[-1] - y

        # calc deltas in backprop
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # reverse deltas so in correct order (we calculated them backwards)
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss

    def plot_loss(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.epoch_data, self.loss_data, "-*")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()
