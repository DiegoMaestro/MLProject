from enum import IntEnum

class Element(IntEnum):
    EMPTY       = 0b0000
    BLACK_MAN   = 0b0101
    BLACK_KING  = 0b0111
    WHITE_MAN   = 0b1001
    WHITE_KING  = 0b1011
    
class Flags(IntEnum):
    VALUE       = 0b0011
    BLACK       = 0b0100
    WHITE       = 0b1000
    KING        = 0b0010
    DIFFERENT   = 0b1100
    
class Direction(IntEnum):
    SOUTHWEST = -9
    SOUTHEAST = -7
    NORTHWEST = 7
    NORTHEAST = 9
    
class Adjacent(IntEnum):
    SOUTHWEST = 0
    SOUTHEAST = 1
    NORTHWEST = 2
    NORTHEAST = 3
    
class Result(IntEnum):
    LOSS = -2
    DRAW = 0
    WIN = 1