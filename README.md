## Machine Learning Project
Run the script [evolutionStrategyProgram.py](#evolutionStrategyProgram) as follows to start the training:

    python evolutionStrategyProgram.py [-l <load path>] [ -s <save path 1> ... <save path N>]

The [constants.py](#constants) file contains all the tunable parameters.
In every save paths the program will, for every generation, create a directory containing the data of the current generation and the data for every network of that generation. The load path needs to be of one of the generations prevoiusly saved. 

Run the script [play.py](#play) as follows to play against a network (from terminal, no interface):

    python play.py <load path>
    
The load path needs to be of one of the networks previously generated and saved.

### <a name="evolutionStrategyProgram" href="#evolutionStrategyProgram">#</a> evolutionStrategyProgram.py - [source](evolutionStrategyProgram.py)
Main python module. Checks if the program was executed with the right parameters (if any), creates the EvolutionstrategyProgram class, and then starts the training. The EvolutionstrategyProgram class contains the neural networks and the parameters and methods for generating the new generations.
For the training, every network will play with a number of other networks (depending on the parameter defined in [contants.py](constants.py)), gaining or losing points according to the match result. When all games have ended, the networks (half of the total) that scored the best will produce an offspring and go to the next generation.

### <a name="neuralNetwork" href="#neuralNetwork">#</a> neuralNetwork.py [source](neuralNetwork.py)
The architecture of the neural networks is 32 x 40 x 10 x 1 and use the hyperbolic tanget as the activation function.

### <a name="constants" href="#constants">#</a> constants.py - [source](constants.py)
Contains the parameters that define the neural network's architecture, the depth of the minmax algorithm, the number of networks, the number of games each network has to play every generation, and the number of generations.

### <a name="player" href="#player">#</a> player.py - [source](player.py)
Wrapper class for the neural network. Contains a [search algorithm] (https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) which, through the use of the neural network, decides the best move.

### <a name="game" href="#game">#</a> game.py - [source](game.py)
Class which handles a game between two networks or between a network and a human.

### <a name="board" href="#board">#</a> board.py - [source](board.py)
Class which implements the checkers board and mechanics.

### <a name="enums" href="#enums">#</a> enums.py - [source](enums.py)
Contains enums used by the Board class.

### <a name="play" href="#play">#</a> play.py - [source](play.py)
Main module for playing against a neural network.
