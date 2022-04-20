import numpy as np


class LinearLayer:
    def __init__(self, n_in, n_out, batch_size, activation=None, lr=0.001):
        self.W = np.random.normal(scale=0.01, size=(n_in, n_out))
        self.b = np.zeros((batch_size, n_out))
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.parameter = {'name':'Linear', 'size':[n_in, n_out], 'activation':activation}

    def forward(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b
        if self.activation is 'relu':
            output = np.maximum(0, output)
        if self.activation is 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        if self.activation is 'tanh':
            output = np.tanh(output)
        self.activated_output = output
        return output

    def backward(self, dout):
        if self.activation is 'relu':
            self.activated_output[self.activated_output <= 0] = 0
            self.activated_output[self.activated_output > 0] = 1
            dout = dout * self.activated_output
        if self.activation is 'sigmoid':
            dout = self.activated_output * (1 - self.activated_output) * dout
        if self.activation is 'tanh':
            dout = (1 - self.activated_output ** 2) * dout
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = dout
        self.W = self.W - self.dW * self.lr / self.batch_size
        self.b = self.b - self.db * self.lr / self.batch_size
        return dx


class SoftMax:
    y_hat = []

    def __init__(self):
        super(SoftMax, self).__init__()
        self.parameter = {'name':'SoftMax'}

    def forward(self, x):
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        self.y_hat = x_exp / partition
        return self.y_hat

    def backward(self, y):
        dout = self.y_hat - y
        return dout


class MLP:
    def __init__(self, input_size, batch_size, num_classes, lr=0.001, hidden_layer_sizes=(), activation='relu'):

        self.layer_list = [[hidden_layer_sizes[i], hidden_layer_sizes[i + 1]]
                           for i in range(len(hidden_layer_sizes) - 1)]
        self.input_layer = LinearLayer(input_size, hidden_layer_sizes[0], batch_size, activation, lr=lr)
        self.classifier = LinearLayer(hidden_layer_sizes[-1], num_classes, batch_size, activation, lr=lr)
        self.softmax = SoftMax()
        self.batch_size = batch_size
        self.lr = lr

        self.layers = [self.input_layer]
        for i in range(len(self.layer_list)):
            self.layers.append(LinearLayer(self.layer_list[i][0], self.layer_list[i][1], batch_size, activation, lr=lr))
        self.layers.append(self.classifier)
        self.layers.append(self.softmax)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def parameter(self):
        for i in range(len(self.layers)):
            print("layer {}: {}".format(i + 1, self.layers[i].parameter))

