import numpy as np
import mnist_loader
import matplotlib.pyplot as plt


def sigmoid_activation(n):
    return 1 / (1 + np.exp(-n))


def derivative_sigmoid(x):
    return x * (1-x)


def mini_batch_maker(data_batch, mini_batch_size):
    mini_batches = []
    mini_batch1 = []
    for i in data_batch:
        mini_batch1.append(i)
        if len(mini_batch1) == mini_batch_size:
            mini_batches.append(mini_batch1)
            mini_batch1 = []
    return mini_batches


class network:
    def __init__(self, net_map):
        n_inputs = net_map[0]
        self.all_layers = []
        for i in net_map[1:]:
            layer = network_layer(i, n_inputs)
            n_inputs = i
            self.all_layers.append(layer)

    def feed_forward(self, inputs):
        outputs = []
        for layer in self.all_layers:
            layer.feed_forward(inputs)
            inputs = layer.output
            outputs.append(layer.output)
        self.output = outputs[-1]

    def back_prop(self, targets):
        last_layer = self.all_layers[-1]
        for i in range(len(last_layer.weights)):
            delta = ((last_layer.output[i] - targets[i]) * 2
                     * derivative_sigmoid(last_layer.output[i]))
            last_layer.B_deltas.append(delta)
            wd = delta * last_layer.inputs
            last_layer.W_deltas.append(wd)
        for i in range(2,len(self.all_layers) + 1):
            layer = self.all_layers[-i]
            front_layer = self.all_layers[-i+1]
            for j in range(len(layer.weights)):
                delta = 0
                for k in range(len(front_layer.B_deltas)):
                    delta += (front_layer.weights[k][j] * 
                              front_layer.B_deltas[k])
                delta *= derivative_sigmoid(layer.output[j])
                layer.B_deltas.append(delta)
                wd = delta * layer.inputs
                layer.W_deltas.append(wd)
        for layer in self.all_layers:
            layer.wdelta_jar.append(layer.W_deltas)
            layer.bdelta_jar.append(layer.B_deltas)
            layer.W_deltas = []
            layer.B_deltas = []
        
        
    def correction(self):
        for layer in self.all_layers:
            layer.correction()

    def train(self, data_batch):
        mini_batch_size = 10
        epochs = 3
        for i in range(epochs):
            np.random.shuffle(data_batch)
            mini_batches = mini_batch_maker(data_batch, mini_batch_size)
            for mini_batch in mini_batches:
                for trainig_eg in mini_batch:
                    self.feed_forward(trainig_eg[0])
                    self.back_prop(trainig_eg[1])
                self.correction()
            print("epoch over")


class network_layer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = []
        self.biases = []
        self.W_deltas = []
        self.B_deltas = []
        self.wdelta_jar = []
        self.bdelta_jar = []
        for i in range(n_neurons):
            self.biases.append(0)
            set_of_weights = []
            for j in range(n_inputs):
                set_of_weights.append(np.random.randn())
            self.weights.append(set_of_weights)

    def feed_forward(self, inputs):
        self.output = sigmoid_activation(
            np.dot(self.weights, inputs) + self.biases)
        self.inputs = np.array(inputs)

    def ultimate_delta(self):
        self.W_deltas = np.mean(self.wdelta_jar, axis=0)
        self.B_deltas = np.mean(self.bdelta_jar, axis=0)
        self.wdelta_jar = []
        self.bdelta_jar = []

    def correction(self):
        learning_rate = 3
        self.ultimate_delta()
        self.W_deltas *= learning_rate
        self.B_deltas *= learning_rate
        self.weights = np.subtract(self.weights,self.W_deltas)
        self.biases = np.subtract( self.biases,self.B_deltas)
        self.W_deltas = []
        self.B_deltas = []


def main():

    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    training_data_list = []
    for x, y in training_data:
        x = np.ravel(x)
        y = np.ravel(y)
        training_data_list.append((x, y))
    
    little = []
    np.random.shuffle(training_data_list)
    for i in range(1000):
        little.append(training_data_list[i])
    
    
    validation_data_list = []
    for x, y in validation_data:
        x = np.reshape(x , (28,28))
        y = np.ravel(y)
        validation_data_list.append((x, y))
    
    
    plt.imshow(validation_data_list[0][0])
    plt.show()

    network1 = network([784 ,30, 10])
    
    v = np.ravel(validation_data_list[0][0])
    network1.feed_forward(v)
    print(network1.output)

    network1.train(training_data_list)
    
    v = np.ravel(validation_data_list[0][0])
    network1.feed_forward(v)
    print(network1.output)


main() # made by pangshanbe
