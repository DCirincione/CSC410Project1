###############
## Matt Lepinski
## Version 1
## Code to support Project 1 for CSC 410 - Fall 2025
###############

## To run a game
##    between the players in Alice.py and Bob.py
##
## python GameEngine.py Alice Bob
##
## By default the output will go into file game0.log
## You can choose a different file as follows
##
## python GameEngine.py Alice Bob X
##
## ... will save the output in gameX.log

import importlib
import sys
import os
import re
import json
import random
import GameRules as Grules

# Load the code for the AI players
# Use command line arguments
# If a player isn't provided use 'DefaultPlayer.py'

players = {}

if ( len(sys.argv) > 1):
    players['Light'] = importlib.import_module(sys.argv[1])
else:
    players['Light'] = importlib.import_module('DefaultPlayer')

if ( len(sys.argv) > 2):
    players['Dark'] = importlib.import_module(sys.argv[2])
else:
    players['Dark'] = importlib.import_module('DefaultPlayer')

if ( len(sys.argv) > 3):
    filename = "game" + sys.argv[3] + ".log"
else:
    filename = "game0.log"

    
################################

def play_game(players,log_name):

    # Randomly swap Light and Dark half the time
    # ... This gives each player a 50/50 chance of being Dark
    if random.randint(1,2) == 1:
        temp = players['Light']
        players['Light'] = players['Dark']
        players['Dark'] = temp
    
    # open logfile for writing
    logfile = open(log_name, 'w')

    nameLight = players['Light'].name()
    nameDark = players['Dark'].name()
    
    logfile.write(f'Light Player is {nameLight} \n Dark Player is {nameDark} \n')
    state = Grules.getInitialState()
    startPlayer = state['Turn']
    gameOver = False

    logfile.write(f"{startPlayer} plays first \n")

    logfile.write("Starting State is:\n")
    logfile.write(json.dumps(state))

    while (not gameOver):
        activePlayer = players[ state['Turn'] ]
        move = activePlayer.getMove(state)

        logfile.write(f"\nMove for {state['Turn']} Player \n")
        logfile.write(json.dumps(move))
        #print(f"... {move}")
        new_state = Grules.playMove(state, move)

        if new_state != None:
            state = new_state
            ## Grules.printState(state)
            gameOver=Grules.isGameOver(state)
            if gameOver:
                state = Grules.endGame(state)
                logfile.write(f"\nGame Ends. Player {state['Turn']} has no legal moves.\n")
                logfile.write(json.dumps(state))
                #print(f"\n   GAME OVER. Player {state['Turn']} has no legal moves.")
                #Grules.printState(state)
                
                if state['LightCapture'] >= state['DarkCapture']:
                    return 'Light'
                else: 
                    return 'Dark'
                
        else:
            gameOver = True
            logfile.write(f"Illegal Move. End of Game \n")
            print(f"Illegal Move {move}. Player {state['Turn']} Forfeits.")

            if state['Turn'] == 'Light':
                return 'Dark'
            else:
                return 'Light'

        
#############################################

# Main Program
if __name__=="__main__":
    play_game(players, filename)

