import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.hidden_size, self.input_size)
        self.W2 = np.random.randn(self.output_size, self.hidden_size)
        self.B1 = np.random.randn(self.hidden_size)
        self.B2 = np.random.randn(self.output_size)
        self.B1.shape += (1,)
        self.B2.shape += (1,)
    
    def get_training_weights(self):
        # Load MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        data = pd.DataFrame(mnist.data, columns=mnist.feature_names)
        np.random.shuffle(data.values)
        m, self.pixels = data.shape

        # Separate dataset into training set and test set
        train_size = int(0.8 * m)  # 80% of the data for training
        test_size = m - train_size  # Remaining 20% for testing

        self.X_train = data.iloc[:train_size].values.T
        self.Y_train = mnist.target[:train_size].values.reshape(1, train_size)

        self.X_test = data.iloc[train_size:].values.T
        self.Y_test = mnist.target[train_size:].values.reshape(1, test_size)

        # Convertir les objets Categorical en tableaux NumPy
        self.Y_train = self.Y_train.codes if isinstance(self.Y_train, pd.Categorical) else self.Y_train
        self.Y_test = self.Y_test.codes if isinstance(self.Y_test, pd.Categorical) else self.Y_test

        self.one_hot_Y()

    def one_hot_Y(self):

        # trained data
        self.correct_Y_train = np.zeros((self.output_size, self.Y_train.size))
        self.correct_Y_train[self.Y_train, np.arange(self.Y_train.size)] = 1

        # test data
        self.correct_Y_test = np.zeros((self.output_size, self.Y_test.size))
        self.correct_Y_test[self.Y_test, np.arange(self.Y_test.size)] = 1

    def ReLu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        # Subtract the maximum value of Z for numerical stability
        max_Z = np.max(Z)
        exp_Z = np.exp(Z - max_Z)
        return exp_Z / np.sum(exp_Z)

    
    def dsoftmax(self, Z):
        s = self.softmax(Z)
        return s * (1 - s)

    def forward_propagation(self, X, Y, nr_correct):
        self.Z1 = self.W1 @ X + self.B1
        self.A1 = self.ReLu(self.Z1)
        
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.softmax(self.Z2)

        nr_correct += int(np.argmax(self.A2) == np.argmax(Y))
        return nr_correct


    def back_propagation(self, X, Y, learning_rate):
        dZ2 = self.A2 - Y
        dW2 = np.dot(dZ2, self.A1.T)
        dB2 = dZ2
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(dZ1, X.T)
        dB1 = dZ1

        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * dB2
        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * dB1

    def train(self, epochs, learning_rate):
        for i in range(epochs):
            nr_correct = 0
            for X, Y in zip(self.X_train.T, self.correct_Y_train.T):
                X.shape += (1,)
                Y.shape += (1,)
                nr_correct = self.forward_propagation(X, Y, nr_correct)
                self.back_propagation(X, Y, learning_rate)
                nr_correct += int(np.argmax(self.A2) == np.argmax(Y))
            accuracy = nr_correct / self.X_train.shape[1] * 100
            print(f'Epoch {i+1}/{epochs}, Training Accuracy: {accuracy:.2f}%')



NN = NeuralNetwork(784, 16, 10)
NN.get_training_weights()
NN.train(3, learning_rate=0.1)

from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

images, labels = get_mnist()
W1 = np.random.uniform(-0.5, 0.5, (50, 784))
W2 = np.random.uniform(-0.5, 0.5, (10, 50))
b1 = np.zeros((50, 1))
b2 = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 5
training_set_size = 40000

training_set = images[:training_set_size]
training_labels = labels[:training_set_size]

testing_set = images[training_set_size:]
testing_labels = labels[training_set_size:]

for epoch in range(epochs):
    for img, l in zip(training_set, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        Z1 = b1 + W1 @ img
        A1 = 1 / (1 + np.exp(-Z1))
        # Forward propagation hidden -> output
        Z2 = b2 + W2 @ A1
        A2 = 1 / (1 + np.exp(-Z2))

        # Cost / Error calculation
        e = 1 / len(A2) * np.sum((A2 - l) ** 2, axis=0)
        nr_correct += int(np.argmax(A2) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_2 = (A2 - l)
        W2 += -learn_rate * delta_2 @ np.transpose(A1)
        W2 += -learn_rate * delta_2
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(W2) @ delta_2 * (A1 * (1 - A1))
        W1 += -learn_rate * delta_h @ np.transpose(img)
        b1 += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    while True:
        try:
            index = int(input(f"Enter a number (0 - {59999 - training_set_size}): "))
            if 0 <= index < 60000 - training_set_size:
                break
        except ValueError:
            print("Invalid input, try again")
    index = int(index)
    img = testing_set[index]
    label = testing_labels[index]

    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    Z1 = b1 + W1 @ img.reshape(784, 1)
    A1 = 1 / (1 + np.exp(-Z1))
    # Forward propagation hidden -> output
    Z2 = b2 + W2 @ A1
    A2 = 1 / (1 + np.exp(-Z2))
    print(sum(A2))
    plt.title(f"Guessing a {A2.argmax()} with {A2} and is a {label.argmax()}: ")
    plt.show()
