import numpy as np
import pickle

global path
# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the neural network class
class NeuralNetwork:
    def __init__(self):
        # Initialize weights randomly with mean 0
        self.synaptic_weights1 = 2 * np.random.random((2, 2)) - 1
        self.synaptic_weights2 = 2 * np.random.random((2, 1)) - 1

    def forward(self, X):
        # Propagate inputs through the network
        self.layer1 = sigmoid(np.dot(X, self.synaptic_weights1))
        self.output = sigmoid(np.dot(self.layer1, self.synaptic_weights2))
        return self.output

    def train(self, X, y, num_iterations, learning_rate):
        # Define hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # Initialize moments and squared moments
        m_dw1 = np.zeros(self.synaptic_weights1.shape)
        m_dw2 = np.zeros(self.synaptic_weights2.shape)
        v_dw1 = np.zeros(self.synaptic_weights1.shape)
        v_dw2 = np.zeros(self.synaptic_weights2.shape)

        # Train the network
        for i in range(num_iterations):
            global path
            # Forward pass
            output = self.forward(X)

            # Compute error
            error = y - output


            # Backpropagation
            delta2 = error * sigmoid_derivative(output)
            delta1 = np.dot(delta2, self.synaptic_weights2.T) * sigmoid_derivative(self.layer1)

            # Compute gradients
            dw2 = np.dot(self.layer1.T, delta2)
            dw1 = np.dot(X.T, delta1)

            # Update moments
            m_dw1 = beta1 * m_dw1 + (1 - beta1) * dw1
            m_dw2 = beta1 * m_dw2 + (1 - beta1) * dw2
            v_dw1 = beta2 * v_dw1 + (1 - beta2) * np.square(dw1)
            v_dw2 = beta2 * v_dw2 + (1 - beta2) * np.square(dw2)

            # Compute bias-corrected moments
            m_dw1_hat = m_dw1 / (1 - beta1 ** (i + 1))
            m_dw2_hat = m_dw2 / (1 - beta1 ** (i + 1))
            v_dw1_hat = v_dw1 / (1 - beta2 ** (i + 1))
            v_dw2_hat = v_dw2 / (1 - beta2 ** (i + 1))

            # Update weights
            self.synaptic_weights1 += learning_rate * m_dw1_hat / (np.sqrt(v_dw1_hat) + eps)
            self.synaptic_weights2 += learning_rate * m_dw2_hat / (np.sqrt(v_dw2_hat) + eps)

            # Print loss during training
            if i % 10000 == 0:
                print("Loss after iteration ", i, ":", np.mean(np.abs(error)))
                path.append(error)


# Define input data and labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

path = [X]
# Create a neural network object and train it
nn = NeuralNetwork()
nn.train(X, y, 100000, 0.01)
"""
filename = 'finalized_model.sav'
pickle.dump(nn, open(filename, 'wb'))
"""




