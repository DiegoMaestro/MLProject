import sys
from evolutionStrategyProgram import EvolutionStrategyProgram
from game import Game

def main():
    if (len(sys.argv) != 2):
        print(f"Usage: {sys.argv[0]} <load path>")
        sys.exit()

    p = EvolutionStrategyProgram(sys.argv[0])
    game = Game(None, p.neuralNetworks[0]);
    game.playGameUser();

if __name__ == "__main__":
    main()
