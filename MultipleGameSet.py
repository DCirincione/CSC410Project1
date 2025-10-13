###############
## Matt Lepinski
## Version 1
## Code to support Project 1 for CSC 410 - Fall 2025
###############

## This code runs a set of games between the same two players
##
## You can change the identify of the players 
##  ... as well as the number of games played
##  ... by altering the constants at the top of this file

## Do not include the ".py", just the name of the file without an extension

FIRST_PLAYER = "minimax" 
SECOND_PLAYER = "MiniMaxPlayer1"
NUM_GAMES = 5

import importlib
import random
import sys
import os
import re
import json
import GameRules as Grules

################################

def play_game(the_players,log_name):
    
    # Randomly swap Light and Dark half the time
    # ... This gives each player a 50/50 chance of being Dark
    who_is_light = random.randint(1,2)

    # By default, FIRST_PLAYER is Light,
    #   So we swap when SECOND_PLAYER is chosen to be Light
    players = {}
    if who_is_light == 2:
        players['Light'] = the_players['Dark']
        players['Dark'] = the_players['Light']
    else:
        players['Light'] = the_players['Light']
        players['Dark'] = the_players['Dark']
    
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
                    return (who_is_light, 'Light')
                else: 
                    return (who_is_light, 'Dark')
                
        else:
            gameOver = True
            logfile.write(f"Illegal Move. End of Game \n")
            print(f"Illegal Move {move}. Player {state['Turn']} Forfeits.")

            if state['Turn'] == 'Light':
                return (who_is_light, 'Dark')
            else:
                return (who_is_light, 'Light')


# Play a set of  Games Between Two AI Players
def play_set(play_1, play_2):

    the_players = {}
    the_players['Light'] = importlib.import_module(play_1)
    the_players['Dark'] = importlib.import_module(play_2)

    print(f"{the_players['Light'].name()} vs {the_players['Dark'].name()}")
    
    first_wins = 0
    second_wins = 0
    
    for num in range(NUM_GAMES):
        print(f"Game {num+1}")
        log_name = f'{play_1}_vs_{play_2}_{num}.log'
        (who_light, winner) = play_game(the_players, log_name)
        print(who_light, winner)

        if (winner == 'Light') and (who_light == 1):
            first_wins += 1
            print(f"{FIRST_PLAYER} is Light and {FIRST_PLAYER} wins")
        elif (winner == 'Dark') and (who_light == 2):
            first_wins += 1
            print(f"{SECOND_PLAYER} is Light and {FIRST_PLAYER} wins")
        else:
            second_wins += 1
            if (who_light == 1):
                print(f"{FIRST_PLAYER} is Light and {SECOND_PLAYER} wins")
            else:
                print(f"{SECOND_PLAYER} is Light and {SECOND_PLAYER} wins")

    print(f"{play_1} Wins {first_wins}")
    print(f"{play_2} Wins {second_wins}")

    return (first_wins, second_wins)

# Main Program
if __name__=="__main__":
    play_set(FIRST_PLAYER, SECOND_PLAYER)


    


