from flask import Flask
from flask import render_template
import sys
import json

import GameRules as Grules

statelist =[]
movelist = []
players = {}

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("watchgame.html")

@app.route('/gamemove/<num>')
def gamemove(num):
    global movelist
    num = int(num)
    if num < 0:
        num = 0
    if num >= len(movelist):
        dict = {}
        dict['Row'] = -1
        dict['Col'] = -1
        dict['Direction'] = 'W'
        return json.dumps(dict)

    return json.dumps(movelist[num])

@app.route('/gamestate/<num>')
def gamestate(num):
    global statelist
    num = int(num)
    if num < 0:
        num = 0
    if num >= len(statelist):
        num = len(statelist) -1 
    
    return json.dumps(statelist[num])

@app.route('/getplayers/')
def getplayers():
    return json.dumps(players)

#Read a logfile to obtain the initial state
def initial_state(filename):
    logfile = open(filename)
    line_num = 0

    for line in logfile:
        if line_num == 4:
            state = json.loads(line)
            return state
        line_num += 1

    return {'Error':'Initial State Not Found'}

    
#Read a logfile to obtain a list of game states
def create_statelist(filename):
    global statelist
    global movelist
    global players

    game_state = initial_state(filename)
    
    logfile = open(filename)

    line_num = 0
    
    for line in logfile:
        if line_num == 0:
            words = line.split()
            players['Light'] = words[3]
        if line_num == 1:
            words = line.split()
            players['Dark'] = words[3]
        if line_num == 2:
            words = line.split()
            game_state['Turn'] = words[0]

        if line[0] == "{":
            dict = json.loads(line)
            
            # Check if it is a Move. Moves have 'Row' entries.
            if 'Row' in dict:
                game_state = Grules.playMove(game_state, dict)
                movelist.append(dict)
            else:
                game_state = dict
            statelist.append(game_state)

        line_num += 1
            


if __name__ == '__main__':
    logfile_name = sys.argv[1]

    create_statelist(logfile_name)

    # ... For debugging ...
    # sys.stderr.write(str(len(movelist)) + "\n")
    # sys.stderr.write(str(len(statelist)) + "\n")

    # for state in statelist:
    #    sys.stderr.write(str(state) + "\n") 
    
    my_port = 5000
    app.run(host='0.0.0.0', port = my_port)
