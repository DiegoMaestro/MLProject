## Machine Learning Project
Run the script [evolutionStrategyProgram.py](#evolutionStrategyProgram) as follows to start the training:

    python evolutionStrategyProgram.py [-l <load path>] [ -s <save path 1> ... <save path N>]

The [constants.py](#constants) file contains all the tunable parameters.

### <a name="evolutionStrategyProgram" href="#evolutionStrategyProgram">#</a> evolutionStrategyProgram.py - [source](evolutionStrategyProgram.py)
Main python module. Checks if the program was executed with the right parameters (if any),creates the EvolutionstrategyProgram class, and then starts the training.



### [neuralNetwork.py](https://github.com/DiegoMaestro/MLProject/neuralNetwork.py)


### <a name="constants" href="#constants">#</a> constants.py - [source](https://github.com/DiegoMaestro/MLProject/constants.py)
Contains the parameters that define the neural network's architecture, the depth of the minmax algorithm, the number of networks, the number of games each network has to play every generation, and the number of generations.
