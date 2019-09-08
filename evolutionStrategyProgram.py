from collections import deque
from math import ceil, floor
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import getopt
import json
import os
import re
import sys
import time

from constants import ALPHA_BETA_DEPTH, NETWORKS_NUMBER, INPUT_LAYER_LENGTH,\
    FIRST_LAYER_LENGTH, SECOND_LAYER_LENGTH, OUTPUT_LAYER_LENGTH, TAU,\
    GAMES_PER_NETWORK, GENERATIONS_PER_RUN
from game import Game
from neuralNetwork import NeuralNetwork

SAVE_PATHS = set()

class EvolutionStrategyProgram(object):
    def __init__(self, path):
        # If no data is loaded, initializes the data from generation 0.
        if path is None or not os.path.isdir(path):
            self.generation = 0
            # The sigmas for every network are all initialized to 0.05.
            self.sigmas = np.array([[
                    np.full((FIRST_LAYER_LENGTH, INPUT_LAYER_LENGTH + 1), 0.05),
                    np.full((SECOND_LAYER_LENGTH, FIRST_LAYER_LENGTH + 1), 0.05),
                    np.full((OUTPUT_LAYER_LENGTH, SECOND_LAYER_LENGTH + 1), 0.05)
                ]
                for n in range(NETWORKS_NUMBER//2)])

            # Half of the networks are initialized with tanh acivation function
            # and half with lReLU. Space is also being allocated for the offsprings.
            self.neuralNetworks = [
                NeuralNetwork(number=number + 1)
                for number
                in range(NETWORKS_NUMBER//2*self.generation, NETWORKS_NUMBER//2*(self.generation + 1))
            ] + [None] * (NETWORKS_NUMBER//2)
        else:
            # Loads data
            self.load(path)


    # Saves the current generation data
    def save(self):
        for savePath in SAVE_PATHS:
            path = Path(savePath) / f"Generation{self.generation}"
            print(f"Saving progress in directory {path}")
            os.makedirs(path)

            sigmas = []
            for network in self.sigmas:
                sigmas.append([[sigma.tolist() for sigma in layer] for layer in network])

            data = {
                    "generation": self.generation,
                    "sigmas": sigmas,
                    "networks": [network.number for network in self.neuralNetworks]
                    }

            with open(path / "EvolutionStrategyProgram.txt", 'x') as outfile:
                json.dump(data, outfile, indent=4)

            for network in self.neuralNetworks:
                network.save(path)

    # Loads data from a previously saved instance
    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        with open(path / "EvolutionStrategyProgram.txt") as json_file:
            data = json.load(json_file)

            self.generation = data["generation"]
            self.sigmas = np.array([[np.array(sigma) for sigma in layer] for layer in data["sigmas"]])
            self.neuralNetworks = [NeuralNetwork(path= path / f"NeuralNetwork{number}.txt") for number in data["networks"]]

    def mutateSigmas(self):
        # sigma' = sigma * e^( tau * N(0,1) )
        self.sigmas *= [[np.exp(TAU * np.random.randn(FIRST_LAYER_LENGTH, INPUT_LAYER_LENGTH + 1)),
                        np.exp(TAU * np.random.randn(SECOND_LAYER_LENGTH, FIRST_LAYER_LENGTH + 1)),
                        np.exp(TAU * np.random.randn(OUTPUT_LAYER_LENGTH, SECOND_LAYER_LENGTH + 1))]
                        for n in range(NETWORKS_NUMBER//2)]

    def sortNetworks(self):
        # Orders the networks based on their fitness and keeps the half better ones.
        # In case of two networks with the same fitness, the younger has priority.
        self.neuralNetworks.sort(key=lambda network: (network.getFitness(), network.getNumber()) if network is not None else (0, 0), reverse=True)
        # todo: remove
        for network in self.neuralNetworks:
            print(f"Number: {network.number}, fitness: {network.fitness}")

    def deleteWorstNetworks(self):
        self.neuralNetworks[NETWORKS_NUMBER//2:] = [None]*(NETWORKS_NUMBER//2)
        for network in self.neuralNetworks[:NETWORKS_NUMBER//2]: network.resetFitness()

    def generateOffsprings(self):
        # The sigma (and the variation) array contains, for each network, the self-adaptive parameters for both weights and biases.
        # with the latter being the last column of the matrix.
        for network, variation, index in zip(self.neuralNetworks[:NETWORKS_NUMBER//2],
                                             self.sigmas * [[np.random.randn(FIRST_LAYER_LENGTH, INPUT_LAYER_LENGTH + 1),
                                                             np.random.randn(SECOND_LAYER_LENGTH, FIRST_LAYER_LENGTH + 1),
                                                             np.random.randn(OUTPUT_LAYER_LENGTH, SECOND_LAYER_LENGTH + 1)]
                                                             for n in range(NETWORKS_NUMBER//2)],
                                             range(NETWORKS_NUMBER//2, NETWORKS_NUMBER)):
            variatedWeights = variation[0][:,:-1], variation[1][:,:-1], variation[2][:,:-1]
            variatedBiases = np.array(variation[0][:,-1])[:,np.newaxis],\
                np.array(variation[1][:,-1])[:,np.newaxis],\
                np.array(variation[2][:,-1])[:,np.newaxis]

            self.neuralNetworks[index] = NeuralNetwork(weights=network.weights + variatedWeights,
                                                       biases=network.biases + variatedBiases,
                                                       number=self.generation*(NETWORKS_NUMBER//2) + index + 1,
                                                       parent=network.number)
        self.generation += 1

    # Creates a list of opposing teams for the tournament (round-robin)
    def getOpponentsCouplesList(self, indices):
        l = list(indices)
        indices.rotate()
        return zip(l[:NETWORKS_NUMBER//2], [NETWORKS_NUMBER - 1] + l[:-NETWORKS_NUMBER//2:-1])

    # For the current generation, plays a tournament where every network plays
    # a predetermined number of games. All the games are played asynchronously.
    def play(self):
        indices = deque(range(NETWORKS_NUMBER - 1))
        p = Pool(processes=NETWORKS_NUMBER//2)

        for iteration in range(GAMES_PER_NETWORK):
            print(f"Playing games number {iteration + 1}...")
            results = [(player1, player2, p.apply_async(Game(self.neuralNetworks[player1], self.neuralNetworks[player2]).playGame))
                       for player1, player2 in self.getOpponentsCouplesList(indices)]
            for player1, player2, points in results:
                points = points.get()
                self.neuralNetworks[player1].updateFitness(points[0])
                self.neuralNetworks[player2].updateFitness(points[1])

    # For every generation, the tournament is played, and some info are displayed.
    # At the end of the execution are displayed the total execution time,
    # a recap of the program configurations, and, if paths were specified at launch,
    # the locations of the saved data
    def evolutionStrategy(self):
        programStart = time.time()

        for generation in range(GENERATIONS_PER_RUN):
            start = time.time()
            print(f"Generating generation {self.generation} offsprings")
            self.mutateSigmas()
            self.deleteWorstNetworks()
            self.generateOffsprings()
            print(f"Generation {self.generation} is now playing")
            self.play()
            end = time.time()
            m, s = divmod(end - start, 60)
            h, m = divmod(m, 60)
            print(f"Generation {self.generation} finished playing in {int(h):02d}:{int(m):02d}:{ceil(s):02d}.")
            self.sortNetworks()
            self.save()
            print()

        programEnd = time.time()
        pm, ps = divmod(programEnd - programStart, 60)
        ph, pm = divmod(pm, 60)
        print("------------------------------------------------\n")
        print(f"Program finished in {int(ph):02d}:{int(pm):02d}:{ceil(ps):02d}.\n\n" +
              "Settings:\n" +
              f"Networks playing per generation: {NETWORKS_NUMBER}\n" +
              f"Games played by each network per generation: {GAMES_PER_NETWORK}\n" +
              f"Depth of alpha-beta pruning search: {ALPHA_BETA_DEPTH}\n" +
              f"Number of generations: {GENERATIONS_PER_RUN}\n" +
              f"Neural Network architecture: {INPUT_LAYER_LENGTH}, {FIRST_LAYER_LENGTH}, {SECOND_LAYER_LENGTH}, {OUTPUT_LAYER_LENGTH}\n\n" +
              "Progresses have been saved at the following locations:")
        for path in SAVE_PATHS: print(path)

# Checks if the save path given is usable
def isPathAvailable(path):
    if not isinstance(path, Path):
        path = Path(path)

    if (os.access(path, os.W_OK)):
        for element in os.listdir(path):
            if re.search('Generation[0-9]+', element) and os.path.isdir(path / element):
                print(f"Directory {element} in path {path} already exists. The program needs to be able to create directories " \
                      "with name \"Generation#\", where # is a positive integer")
                return False
        return True

    print(f"Directory {path} does not exists or the program doesn't have write permission")
    return False

# Gets the paths for loading and/or saving and executes the main program
def main():
    if (len(sys.argv) == 1):
        if (input("If you don't specify a path, no progress will be saved. Do you want to continue? (y\\n)\n") == 'n'):
            print(f"Usage: {sys.argv[0]} [-l <load path>] [ -s <save path 1> ... <save path N>]")
            sys.exit()
    else:
        loadPath = None
        try:
            options, args = getopt.getopt(sys.argv[1:], 'l:s', ['load','save'])
            for opt, value in options:
                if opt in ('-s', '--save'):
                    for path in args:
                        if not isPathAvailable(path):
                            sys.exit()
                        SAVE_PATHS.add(path)
                if opt in ('-l', '--load'):
                    loadPath = value

        except getopt.GetoptError as err:
            print(err)
            print(f"Usage: {sys.argv[0]} [-l <load path>] [ -s <save path 1> ... <save path N>]")
            sys.exit()

    p = EvolutionStrategyProgram(loadPath)
    p.evolutionStrategy()

if __name__ == "__main__":
    main()
