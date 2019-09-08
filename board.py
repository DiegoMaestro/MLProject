from enums import Element, Flags, Direction, Adjacent
from player import Player
import numpy as np
from neuralNetwork import NeuralNetwork
from constants import  INPUT_LAYER_LENGTH, FIRST_LAYER_LENGTH, SECOND_LAYER_LENGTH,\
    OUTPUT_LAYER_LENGTH, ARCHITECTURE


class Board(object):
    possibleMoves = {    0: (None, None, None, 4),   1: (None, None, 4, 5),      2: (None, None, 5, 6),      3: (None, None, 6,7),
                         4: (0, 1, 8, 9),            5: (1, 2, 9, 10),           6: (2, 3, 10, 11),          7: (3, None, 11, None),
                         8: (None, 4, None, 12),     9: (4, 5, 12, 13),         10: (5, 6, 13, 14),         11: (6, 7, 14, 15),
                        12: (8, 9, 16, 17),         13: (9, 10, 17, 18),        14: (10, 11, 18, 19),       15: (11, None, 19, None),
                        16: (None, 12, None, 20),   17: (12, 13, 20, 21),       18: (13, 14, 21, 22),       19: (14, 15, 22, 23),
                        20: (16, 17, 24, 25),       21: (17, 18, 25, 26),       22: (18, 19, 26, 27),       23: (19, None, 27, None),
                        24: (None, 20, None, 28),   25: (20, 21, 28, 29),       26: (21, 22, 29, 30),       27: (22, 23, 30, 31),
                        28: (24, 25, None, None),   29: (25, 26, None, None),   30: (26, 27, None, None),   31: (27, None, None, None) }

    possibleCaptures = { 0: (9,),                    1: (8, 10),                 2: (9,11),                  3: (10,),
                         4: (13,),                   5: (12, 14),                6: (13, 15),                7: (14,),
                         8: (1, 17),                 9: (0, 2, 16, 18),         10: (1, 3, 17, 19),         11: (2, 18),
                        12: (5, 21),                13: (4, 6, 20, 22),         14: (5, 7, 21, 23),         15: (6, 22),
                        16: (9, 25),                17: (8, 10, 24, 26),        18: (9, 11, 25, 27),        19: (10, 26),
                        20: (13, 29),               21: (12, 14, 28, 30),       22: (13, 15, 29, 31),       23: (14, 30),
                        24: (17,),                  25: (16, 18),               26: (17, 19),               27: (18,),
                        28: (21,),                  29: (20, 22),               30: (21, 23),               31: (22,) }

    def __init__(self, board = None):
        if (board is None):
            self.board = [Element.BLACK_MAN] * 12 + [Element.EMPTY] * 8 + [Element.WHITE_MAN] * 12
        else:
            self.board = board

    def copy(self):
        return Board(self.board.copy())

    def reverse(self):
        self.board.reverse()

    def getElement(self, pos):
        return self.board[pos]

    def isElementEmpty(self, pos):
        return self.getElement(pos) is Element.EMPTY

    def isElementPlayerKing(self, pos, player):
        return self.getElement(pos) & (player | Flags.KING)

    def isDifferentPlayer(self, element1, element2):
        return bool(element1 >> 2 ^ element2 >> 2)

    def getPossibleCaptures(self, pos):
        return self.possibleCaptures[pos]

    def getLegalMoves(self, pos, player):
        moves = self.possibleMoves[pos]

        return tuple(filter(lambda np: np is not None and self.isLegalMove((pos, np), player), moves))

    def getLegalCaptures(self, pos, player):
        captures = self.getPossibleCaptures(pos)

        return tuple(filter(lambda np: np is not None and self.isLegalCapture((pos, np), player), captures))

    def getAllLegalMoves(self, player):
        indicesPlayer = [i for i, x in enumerate(self.board) if x & player]

        return [(k,m) for k in indicesPlayer for m in self.getLegalCaptures(k, player)], [(k,m) for k in indicesPlayer for m in self.getLegalMoves(k, player)]

    def getPossibleMoves(self, pos):
        return self.possibleMoves[pos]

    def isInPossibleMoves(self, move):
        return move[1] in self.getPossibleMoves(move[0])

    def isInPossibleCaptures(self, move):
        return move[1] in self.getPossibleCaptures(move[0])

    def getAdjacentPosition(self, pos, adjacent):
        return self.getPossibleMoves(pos)[adjacent.value]

    def getCapturedPosition(self, move):
        if (self.isInPossibleCaptures(move)):
            return self.getAdjacentPosition(move[0], Adjacent[Direction(move[1] - move[0]).name])

    def isLegalMove(self, move, player):
        compare = self.getCompare(player)

        return (self.isInPossibleMoves(move) and self.isElementEmpty(move[1])
                and (compare(move) or bool(self.getElement(move[0]) & Flags.KING)))

    def isLegalCapture(self, move, player):
        capturedPos = self.getCapturedPosition(move)
        compare = self.getCompare(player)

        return (self.isInPossibleCaptures(move) and self.isDifferentPlayer(self.getElement(move[0]), self.getElement(capturedPos))
                and self.isElementEmpty(move[1]) and not self.isElementEmpty(capturedPos) and (compare(move) or bool(self.getElement(move[0]) & Flags.KING)))

    def move(self, move, player):
        if(move == ()):
            return False
        if (self.isLegalMove(move, player)):
            self.board[move[0]], self.board[move[1]] = self.board[move[1]], self.board[move[0]]

            if ((self.getElement(move[1]) is Element.BLACK_MAN and move[1] > 27) or
                (self.getElement(move[1]) is Element.WHITE_MAN and move[1] < 4)):
                self.board[move[1]] = Element(self.board[move[1]] | Flags.KING)
            return False
        elif (self.isLegalCapture(move, player)):
            self.board[move[0]], self.board[move[1]] = self.board[move[1]], self.board[move[0]]
            self.board[self.getCapturedPosition(move)] = Element.EMPTY

            if ((self.getElement(move[1]) is Element.BLACK_MAN and move[1] > 27) or
                (self.getElement(move[1]) is Element.WHITE_MAN and move[1] < 4)):
                self.board[move[1]] = Element(self.board[move[1]] | Flags.KING)
            return True
        return False;

    def getCompare(self, player):
        if (player is Flags.WHITE):
                return lambda move: move[0] > move[1]
        elif (player is Flags.BLACK):
            return lambda move: move[0] < move[1]

    def boardAsValue(self, player):
        boardValue = np.array(self.board)[:, np.newaxis]
        opponent = np.where(boardValue & Flags.DIFFERENT ^ player)
        boardValue &= Flags.VALUE
        boardValue[opponent] *= -1

        return boardValue

    def printBoard(self):
        b = self.board.copy()

        for i in range(len(b)):
            if (b[i] is Element.WHITE_MAN): b[i] = "w"
            elif (b[i] is Element.WHITE_KING): b[i] = "W"
            elif (b[i] is Element.BLACK_MAN): b[i] = "b"
            elif (b[i] is Element.BLACK_KING): b[i] = "B"
            elif (b[i] is Element.EMPTY): b[i] = " "

        i = 12;

        print(f"  {b[i+16]}    {b[i+17]}    {b[i+18]}    {b[i+19]}"+\
            f"\t\t  {i+16}    {i+17}    {i+18}    {i+19}")
        print(f"{b[i+12]}    {b[i+13]}    {b[i+14]}    {b[i+15]}"+\
            f"\t\t{i+12}    {i+13}    {i+14}    {i+15}")
        print(f"  {b[i+8]}    {b[i+9]}    {b[i+10]}    {b[i+11]}  "+\
            f"\t\t  {i+8}    {i+9}    {i+10}    {i+11}  ")
        print(f"{b[i+4]}    {b[i+5]}    {b[i+6]}    {b[i+7]}"+\
            f"\t\t{i+4}    {i+5}    {i+6}    {i+7}")
        print(f"  {b[i]}    {b[i+1]}    {b[i+2]}    {b[i+3]}  "+\
            f"\t\t  {i}    {i+1}    {i+2}    {i+3}  ")

        i = 0;

        print(f"{b[i+8]}    {b[i+9]}    {b[i+10]}    {b[i+11]}  "+\
            f"\t\t {i+8}     {i+9}    {i+10}    {i+11}  ")
        print(f"  {b[i+4]}    {b[i+5]}    {b[i+6]}    {b[i+7]}"+\
            f"\t\t   {i+4}     {i+5}     {i+6}     {i+7}")
        print(f"{b[i]}    {b[i+1]}    {b[i+2]}    {b[i+3]}  "+\
            f"\t\t {i}     {i+1}     {i+2}     {i+3}  ")

    def endGame(self, player):
        playerCaptures, playerMoves = self.getAllLegalMoves(player)

        if (len(playerCaptures) is 0 and len(playerMoves) is 0):
            return True
        return False
