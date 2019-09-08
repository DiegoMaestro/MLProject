from random import shuffle
from cmath import inf
from enums import Flags, Result
import numpy as np

class Player(object):
    def __init__(self, board, player, depth, neuralNetwork):
        self.board = board
        self.player = player
        self.depth = depth if depth % 2 == 0 else depth + 1
        self.neuralNetwork = neuralNetwork

    def endGame(self, winningPlayer):
        if (winningPlayer is None):
            self.neuralNetwork.updateFitness(Result.DRAW)
        elif (self.player & winningPlayer):
            self.neuralNetwork.updateFitness(Result.WIN)
        elif ((self.player ^ Flags.DIFFERENT) & winningPlayer):
            self.neuralNetwork.updateFitness(Result.LOSS)

    def decideMove(self, capturing = None):
        boardState = self.board.copy()

        value, move = self.alphabeta(boardState, self.depth, -inf, inf, True, capturing)
        #value, move = self.minmax(boardState, self.depth, True, capturing)

        return move

    def minmax(self, boardState, depth, maximizingPlayer, capturing = None):
        # The player is the maximizing one while the opponent is the minimizing one
        currentPlayer = self.player if maximizingPlayer else Flags(self.player ^ Flags.DIFFERENT)
        allCaptures, allMoves = boardState.getAllLegalMoves(currentPlayer)

        # If the player already captured an opponent's piece, he could be able to capture another one with the same piece,
        # so the possible captures need to be restricted at the ones involving the player's piece
        allCaptures = [capture for capture in allCaptures if (capture[0] == capturing)] if capturing is not None else allCaptures
        shuffle(allCaptures)
        shuffle(allMoves)

        # A player is obligated to capture if he can
        canCapture = capturing is not None or bool(allCaptures)
        if not canCapture:
            possibleMoves = allMoves
            d = depth - 1
        else:
            possibleMoves = allCaptures
            d = depth

        # depth is 0 or the current player captured a piece and can capture no more or the current player has no available move
        # The board is always evaluated from the player perspective: the player will try to get the highest score (maximizing)
        # and the opponent will try to get the lowest score (minimizing), which is the highest from his point of view
        if (depth == 0 or (capturing is not None and not allCaptures) or (capturing is None and not allMoves)):
            return self.neuralNetwork.feedForward(boardState.boardAsValue(self.player)), ()

        if (maximizingPlayer):
            value = -inf
            bestMove = ()

            for move in possibleMoves:
                newBoardState = boardState.copy()
                newBoardState.move(move, currentPlayer)
                minmax, m = self.minmax(newBoardState, d, canCapture, capturing if not canCapture else move[1])

                if (minmax > value):
                    value = minmax
                    bestMove = move

            return value, bestMove
        else:
            value = +inf

            for move in possibleMoves:
                newBoardState = boardState.copy()
                newBoardState.move(move, currentPlayer)
                minmax, m = self.minmax(newBoardState, d, not canCapture, capturing if not canCapture else move[1])
                value = min(value, minmax)

            return value, m

    def alphabeta(self, boardState, depth, alpha, beta, maximizingPlayer, capturing = None):
        # The player is the maximizing one while the opponent is the minimizing one
        currentPlayer = self.player if maximizingPlayer else Flags(self.player ^ Flags.DIFFERENT)
        allCaptures, allMoves = boardState.getAllLegalMoves(currentPlayer)

        # If the player already captured an opponent's piece, he could be able to capture another one with the same piece,
        # so the possible captures need to be restricted at the ones involving the player's piece
        allCaptures = [capture for capture in allCaptures if (capture[0] == capturing)] if capturing is not None else allCaptures
        shuffle(allCaptures)
        shuffle(allMoves)

        # A player is obligated to capture if he can
        canCapture = capturing is not None or bool(allCaptures)
        if not canCapture:
            possibleMoves = allMoves
            d = depth - 1
        else:
            possibleMoves = allCaptures
            d = depth


        # depth is 0 or the current player captured a piece and can capture no more or the current player has no available move
        # The board is always evaluated from the player perspective: the player will try to get the highest score (maximizing)
        # and the opponent will try to get the lowest score (minimizing), which is the highest from his point of view
        if (depth == 0 or (capturing is not None and not allCaptures) or (capturing is None and not allMoves)):
            return self.neuralNetwork.feedForward(boardState.boardAsValue(self.player)), ()
        if (maximizingPlayer):
            value = -inf
            bestMove = ()

            for move in possibleMoves:
                newBoardState = boardState.copy()
                newBoardState.move(move, currentPlayer)
                alphaBetaValue, m = self.alphabeta(newBoardState, d, alpha, beta, canCapture, capturing if not canCapture else move[1])
                if (alphaBetaValue > value):
                    value = alphaBetaValue
                    bestMove = move
                alpha = max(alpha, value)

                if alpha >= beta:
                        break # beta cut-off
            return value, bestMove
        else:
            value = +inf

            for move in possibleMoves:
                newBoardState = boardState.copy()
                newBoardState.move(move, currentPlayer)
                alphaBetaValue, m = self.alphabeta(newBoardState, d, alpha, beta, not canCapture, capturing if not canCapture else move[1])
                value = min(value, alphaBetaValue)
                beta = min(beta, value)

                if alpha >= beta:
                        break # beta cut-off
            return value, m
