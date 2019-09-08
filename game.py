from player import Player
from board import Board
from enums import Element, Flags, Result
from constants import ALPHA_BETA_DEPTH
import re
import tkinter

class Game(object):
    def __init__(self, neuralNetwork1, neuralNetwork2):
        self.board = board = Board([Element.BLACK_MAN] * 12 + [Element.EMPTY] * 8 + [Element.WHITE_MAN] * 12)
        self.blackPlayer = Player(board, Flags.BLACK, depth=ALPHA_BETA_DEPTH, neuralNetwork=neuralNetwork1)
        self.whitePlayer = Player(board, Flags.WHITE, depth=ALPHA_BETA_DEPTH, neuralNetwork=neuralNetwork2)

        self.turnsWithoutCapture = 0

    def playGame(self):
        currentPlayer = self.blackPlayer

        while (not self.board.endGame(currentPlayer.player) and self.turnsWithoutCapture < 100):
            self.turnsWithoutCapture += 1
            move = currentPlayer.decideMove()

            while (self.board.move(move, currentPlayer.player)):
                self.turnsWithoutCapture = 0
                move = currentPlayer.decideMove(capturing=move[1])

            currentPlayer = self.whitePlayer if currentPlayer is self.blackPlayer else self.blackPlayer

        if self.turnsWithoutCapture > 99: return Result.DRAW, Result.DRAW
        elif currentPlayer is self.blackPlayer: return Result.LOSS, Result.WIN
        else: return Result.WIN, Result.LOSS

    def playGamePrint(self):
        currentPlayer = self.blackPlayer

        while not self.board.endGame(currentPlayer.player):
            self.turnsWithoutCapture += 1
            move = currentPlayer.decideMove()
            print("------------ {} turn -----------------".format(Flags(currentPlayer.player).name))
            print("{}: {} ----- {}".format(Flags(currentPlayer.player).name, move, self.board.getAllLegalMoves(currentPlayer.player)))
            while (self.board.move(move, currentPlayer.player)):
                self.turnsWithoutCapture = 0
                self.board.printBoard()
                move = currentPlayer.decideMove(capturing=move[1])
                print("\n-------------------------------------")
                print("{}: {} ----- {}".format(Flags(currentPlayer.player).name, move, self.board.getAllLegalMoves(currentPlayer.player)))
            self.board.printBoard()

            currentPlayer = self.whitePlayer if currentPlayer is self.blackPlayer else self.blackPlayer
            print("-----------------------------\n\n")

        if self.turnsWithoutCapture > 99: return Result.DRAW, Result.DRAW
        elif currentPlayer is self.blackPlayer: return Result.LOSS, Result.WIN
        else: return Result.WIN, Result.LOSS

    def playGameUser(self):
        user = User(self.board, Flags.BLACK)
        currentPlayer = user

        while not self.board.endGame(currentPlayer.player):
            self.board.printBoard()
            self.turnsWithoutCapture += 1
            move = currentPlayer.decideMove()
            print("------------ {} turn -----------------".format(Flags(currentPlayer.player).name))
            print("{}: {} ----- {}".format(Flags(currentPlayer.player).name, move, self.board.getAllLegalMoves(currentPlayer.player)))
            while (self.board.move(move, currentPlayer.player)):
                self.turnsWithoutCapture = 0
                self.board.printBoard()
                move = currentPlayer.decideMove(capturing=move[1])
                print("\n-------------------------------------")
                print("{}: {} ----- {}".format(Flags(currentPlayer.player).name, move, self.board.getAllLegalMoves(currentPlayer.player)))

            currentPlayer = self.whitePlayer if currentPlayer is user else user
            print("-----------------------------\n\n")

        winner = None if self.turnsWithoutCapture > 99 else Flags.WHITE if currentPlayer is user else Flags.BLACK
        print(winner)

class User(object):
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def decideMove(self, capturing=None):
        allCaptures, allMoves = self.board.getAllLegalMoves(self.player)

        if (capturing or len(allCaptures) > 0):
            if (len(allCaptures) == 0):
                return ()
            while True:
                choice = input("Insert move between the (from, to) pairs:\n{} (captures)".format((allCaptures if capturing is None else [capture for capture in allCaptures if (capture[0] == capturing)])))
                choice = choice.strip()

                if (re.compile('[0-9]{1,2} [0-9]{1,2}').match(choice) is not None):
                    move = tuple(int(pos.strip()) for pos in choice.split(' '))

                    if (move != () and move in allCaptures):
                        print(move)
                        return move
                print("Mossa {} non valida".format(move))
        else:
            while True:
                choice = input("Insert move between the (from, to) pairs:\n{} (moves)".format(allMoves))
                choice = choice.strip()

                if (re.compile('[0-9]{1,2} [0-9]{1,2}').match(choice) is not None):
                    move = tuple(int(pos.strip()) for pos in choice.split(' '))
                    if (move != () and move in allMoves):
                        print(move)
                        return move
                print("Mossa {} non valida".format(move))
