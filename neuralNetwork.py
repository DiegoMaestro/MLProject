from math import tanh
from pathlib import Path
import json
import numpy as np
from constants import ARCHITECTURE

class NeuralNetwork(object):
    def __init__(self,weights=None, biases=None, number=None, parent=None, path=None):
        if (path is None):
            self.number = number
            self.parent = parent
            if (weights is None or biases is None):
                self.biases = np.array([np.random.randn(y, 1) for y in ARCHITECTURE[1:]])
                self.weights = np.array([np.random.randn(x, y) * np.sqrt(2 / (x + y)) for x, y in zip(ARCHITECTURE[1:], ARCHITECTURE[:-1])])
            else:
                self.biases = biases
                self.weights = weights

            self.fitness = 0
        else:
            self.load(path)


    def feedForward(self, x):
        sigma = np.vectorize(lambda z: tanh(z))

        for w, b in zip(self.weights, self.biases):
            x = sigma(w @ x + b)
        return x

    def getNumber(self):
        return self.number

    def getFitness(self):
        return self.fitness

    def updateFitness(self, result):
        self.fitness += result

    def resetFitness(self):
        self.fitness = 0

    def save(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        data = {
                "number": self.number,
                "parent": self.parent,
                "weights": [weight.tolist() for weight in self.weights],
                "biases": [bias.tolist() for bias in self.biases],
                "fitness": self.fitness
                }

        with open(path / f"NeuralNetwork{self.number}.txt", 'x') as outfile:
            json.dump(data, outfile, indent=4)
            return True

    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        with open(path) as json_file:
            data = json.load(json_file)

            self.number = data["number"]
            self.parent = data["parent"]
            self.weights = np.array([np.array(weigth) for weigth in data["weights"]])
            self.biases = np.array([np.array(weight) for weight in data["biases"]])
            self.fitness = data["fitness"]
