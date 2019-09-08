import numpy as np

ALPHA_BETA_DEPTH = 4

NETWORKS_NUMBER = 4
GAMES_PER_NETWORK = NETWORKS_NUMBER - 1
GENERATIONS_PER_RUN = 1

INPUT_LAYER_LENGTH = 32
FIRST_LAYER_LENGTH = 40
SECOND_LAYER_LENGTH = 10
OUTPUT_LAYER_LENGTH = 1

ARCHITECTURE = np.array([INPUT_LAYER_LENGTH, FIRST_LAYER_LENGTH, SECOND_LAYER_LENGTH, OUTPUT_LAYER_LENGTH])

WEIGHTS_AND_BIASES_NUMBER = (INPUT_LAYER_LENGTH + 1)*FIRST_LAYER_LENGTH \
                            + (FIRST_LAYER_LENGTH + 1)*SECOND_LAYER_LENGTH \
                            + (SECOND_LAYER_LENGTH + 1)*OUTPUT_LAYER_LENGTH

TAU = 1 / np.sqrt(2*np.sqrt(WEIGHTS_AND_BIASES_NUMBER))