## This is an example player who makes moves randomly

import GameRules
import random

def name():
    return 'Default Player'

# Returns a random legal move 
def getMove(state):
    legal_moves = GameRules.getAllLegalMoves(state)
    move = random.choice(legal_moves)
    return move

