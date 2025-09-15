## This is an example player who always captures when she is able to capture

import GameRules
import random

def name():
    return 'Alice'

# Returns a random legal move 
def getMove(state):
    legal_moves = GameRules.getAllLegalMoves(state)

    # If there is a capturing move, make it
    for mv in legal_moves:
        if mv['Direction'] in ['N', 'E', 'S', 'W']:
            return mv

    # Otherwise return a random move
    move = random.choice(legal_moves)
    return move
