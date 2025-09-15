import random

# Return the initial state
# Blank board and random player going first
#
# Board Representation: The board is a list of 36 numbers
#   The first 6 numbers represent the top row of the board
#   The next 6 numbers represent the second row from the top
#   ... and so on
#
# Each of these numbers indicates the number of game pieces in
#    the corresponding square
#
# The square in the upper-left [ square 0,0 ] is Dark
# One of Dark's pieces is randomly teleported before the game begins
# Then Light goes first 

def getInitialState():
    board = []
    for i in range(36):
        board.append(1)

    state = {}
    state['Turn'] = 'Light'
    state['LightCapture'] = 0
    state['DarkCapture'] = 0

    square1, square2 = twoRandomDarkSq()
    setPieces(board, square1['row'], square1['col'], 0)
    setPieces(board, square2['row'], square2['col'], 2)

    state['Board'] = board
    
    return state

# Selects two random Dark Squares. Needed to setup the game
def twoRandomDarkSq():
    r1 = random.randint(0,5)
    c1 = random.randint(0,5)
    while color(r1,c1) != 'Dark':
        r1 = random.randint(0,5)
        c1 = random.randint(0,5)

    r2 = random.randint(0,5)
    c2 = random.randint(0,5)
    while (color(r2,c2) != 'Dark') or ((r1 == r2) and (c1 == c2)):
        r2 = random.randint(0,5)
        c2 = random.randint(0,5)

    return ({'row':r1, 'col':c1}, {'row':r2, 'col':c2})

# Does a deep copy of a state to produce a duplicate state
# Note: In particular, this creates a new board that can be modified
#   without changes the original board
def copyState(state):
    new_state = {}
    new_state['Turn'] = state['Turn']
    new_state['LightCapture'] = state['LightCapture']
    new_state['DarkCapture'] = state['DarkCapture']
    new_board = []
    for num in state['Board']:
        new_board.append(num)
    new_state['Board'] = new_board
    return new_state

# Prints the State in a human-readable format
def printState(state):
    brd = state['Board']
    print("")
    for i in range(6):
        row_string = "|"
        for j in range(6):
            if color(i,j) == 'Light':
                row_string = row_string + " " + str(getPieces(brd,i,j)) + " |"
            else:
                row_string = row_string + "*" + str(getPieces(brd,i,j)) + "*|"
        print(row_string)
    print(f"Captures: {state['LightCapture']} Light and {state['DarkCapture']} Dark")
    print(f"{state['Turn']} Player Turn")


# Returns whether a square is Light or Dark
def color(row, col):
    if (row + col)%2 == 0:
        return 'Dark'
    else:
        return 'Light'

# Returns the number of pieces are on the specified square (row, col)
#   for the specified board
def getPieces(board, row, col):
    num = col + row * 6
    return board[num]

# Changes the number of pieces in the specified square (row, col)
#   for the specified board
# count is the new number of pieces that will be in the square
#   after the function is executed
def setPieces(board, row, col, count):
    num = col + row * 6
    board[num] = count

# Returns True if the game is over and False otherwise
# The game ends when the current player has no legal moves
def isGameOver(state):
    legal_moves = getAllLegalMoves(state)
    if len(legal_moves) == 0:
        return True
    else:
        return False

# When the game ends, the state is updated as follows:
#   The player who made the last legal move
#   gets to move the pieces from one of their own square with the most pieces
#     into their own capture area
# This provides a bonus to the player who did not run out of moves
def endGame(state):
    new_state = copyState(state)
    new_brd = new_state['Board']
    
    if new_state['Turn'] == 'Light':
        previous = 'Dark'
    else:
        previous = 'Light'

    most_pieces = -1
    most_r = -1
    most_c = -1

    # Find the square for the previous player that has the most pieces
    for r in range(6):
        for c in range(6):
            if color(r,c) == previous:
                if getPieces(new_brd, r, c) > most_pieces:
                    most_pieces = getPieces(new_brd, r, c)
                    most_r = r
                    most_c = c

    setPieces(new_brd, most_r, most_c, 0)
    if previous == 'Light':
        new_state['LightCapture'] += most_pieces
    else:
        new_state['DarkCapture'] += most_pieces

    return new_state
                
    

# Return a list of legal directions that the active player can move
#  starting in square specified by rowNum, colNum
#
# Note: The pieces in square rowNum, colNum are the pieces being moved
#   If the move is diagonal then these pieces are being moved to the
#     first occupied square along the diagonal
#
#   If the move is horizontal or vertical then the pieces are being
#     dropped one my one in a line to capture one or more opposing squares
#
# Moves are represented as dictionaries with three keys:
#    'Row', 'Col', and 'Direction'
# The 'Row' and 'Col' entries are provided as integers
# The 'Direction' entry is one of the following strings:
#   'N', 'S', 'E', 'W' for Capture
#  'NE', 'NW', 'SE', 'SW' are the diagonal moves to combine stacks of pieces
#
# Note: Row 0 is the row that is furthest North ('N')
#       Col 0 is the column that is furthest West ('W')
#       Row 5 is the row that is furthest South ('S')
#       Col 5 is the column that is furthest East ('E')
def getLegalMoves( state, rowNum, colNum):
    legal_moves =[]
    
    # Process the possible diagonal moves
    moves = getDiagMoves( rowNum, colNum)
    for m in moves:
        if isLegal(state, m):
            legal_moves.append(m)

    # Process the possible capture moves
    moves = getCaptureMoves( rowNum, colNum)
    for m in moves:
        if isLegal(state, m):
            legal_moves.append(m)

    return legal_moves

# Gets a complete list of Legal moves
#    by looping through all possible starting squares
def getAllLegalMoves(state):
    legal_moves = []
    for i in range(6):
        for j in range(6):
            legal_moves.extend( getLegalMoves(state, i, j) )
    return legal_moves

# Returns a list of the 4 possible diagonal moves starting at square (r,c)
def getDiagMoves( r, c):
    moves = []
    
    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'NW'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'NE'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'SW'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'SE'
    moves.append(m)

    return moves

# Returns a list of the 4 possible capturing  moves starting at square (r,c)
def getCaptureMoves( r, c):
    moves = []
    
    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'N'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'E'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'S'
    moves.append(m)

    m = {}
    m['Row'] = r
    m['Col'] = c
    m['Direction'] = 'W'
    moves.append(m)

    return moves
    
# Takes a state and a move (each represented as Dictionaries, see above)
# Returns a new state that results from performing the specified move
#   starting at the specified state
#
# If the move is NOT a legal move, then the function returns None
def playMove(state, move, debug=False):
    row = move['Row']
    col = move['Col']
    # You must start your move on a valid square
    if (row > 5) or (row < 0):
        return None
    if (col > 5) or (col < 0):
        return None

    # You can not move pieces in your opponents' squares
    if state['Turn'] != color(row, col):
        return None
    
    # Diagonal directions are two letters, Captures are one letter
    if len(move['Direction']) == 2:
        return playDiagMove(state, move,debug)
    else:
        return playCaptureMove(state, move,debug)

# Returns True is a move is legal from a given state and False otherwise
def isLegal(state, move):
    if playMove(state, move) == None:
        return False
    else:
        return True

# Creates a new state by making a Diagonal move to combine stacks of pieces
#   ... returns None if the move is not legal
def playDiagMove(state, move, debug=False):
    row = move['Row']
    col = move['Col']
    direct = move['Direction']
    new_state = copyState(state)
    new_board = new_state['Board']

    # You cannot make a move if there are no pieces on the starting square
    count = getPieces(new_board, row, col)
    if count == 0:
        return None

    # Determine the square that the pieces will move to
    target = getTarget(new_board, row, col, direct)
    if target == None:
        return None
    (new_r, new_c) = target

    # Combine the stack of pieces at (row, col) and (new_r, new_c)
    setPieces(new_board, row, col, 0)
    target_count = getPieces(new_board, new_r, new_c)
    setPieces(new_board, new_r, new_c, count + target_count)
    
    # Change whose turn it is
    if new_state['Turn'] == 'Light':
        new_state['Turn'] = 'Dark'
    else:
        new_state['Turn'] = 'Light'

    return new_state

# Determines which square pieces will move to along a diagonal
#   ... empty spaces are skipped
#   ... pieces move to the first non-empty square along the diagonal
def getTarget(board, r, c, direct):
    if direct == 'NW':
        drow = -1
        dcol = -1
    elif direct == 'NE':
        drow = -1
        dcol = 1
    elif direct == 'SW':
        drow = 1
        dcol = -1
    elif direct == 'SE':
        drow = 1
        dcol = 1
    else:
        return None

    done = False
    new_r = r
    new_c = c
    while not done:
        new_r += drow
        new_c += dcol
        if (new_r < 0) or (new_r > 5):
            return None
        if (new_c < 0) or (new_c > 5):
            return None
        if getPieces(board, new_r, new_c) != 0:
            done = True

    return (new_r, new_c)


# Creates a new state by making a Capturing move
#   in one of the four cardinal directions
#   ... returns None if the move is not legal
def playCaptureMove(state, move, debug=False):
    row = move['Row']
    col = move['Col']
    direct = move['Direction']
    new_state = copyState(state)
    new_board = new_state['Board']

    # You cannot make a move if there are no pieces on the starting square
    stack_size = getPieces(new_board, row, col)
    if stack_size == 0:
        return None

    # In order to capture, there must be an opposing (smaller) stack
    #   that you can hit by dropping out your pieces
    #   in the specified direction
    if not canCapture(new_board, row, col, direct, stack_size):
        return None

    # Now we know we can capture and so we change the state
    new_state = doCapture(new_state, row, col, direct)
    
    # Change whose turn it is
    if new_state['Turn'] == 'Light':
        new_state['Turn'] = 'Dark'
    else:
        new_state['Turn'] = 'Light'

    return new_state

# Determines whether there is a (smaller) opposing stack 
#     that can be reached given a starting position and a direction
# Returns True if the capturing move will hit a smaller opposing stack
def canCapture(brd, r, c, direct, init_stack):
    if direct == 'N':
        d_row = -1
        d_col = 0
    elif direct == 'S':
        d_row = 1
        d_col = 0
    elif direct == 'E':
        d_row = 0
        d_col = 1
    elif direct == 'W':
        d_row = 0
        d_col = -1
    else:
        return False

    done = False
    new_r = r
    new_c = c
    count = 0
    while not done:
        count += 1
        new_r += d_row
        new_c += d_col
        if (new_r < 0) or (new_r > 5):
            return False
        if (new_c < 0) or (new_c > 5):
            return False
        if count > init_stack:
            return False
        if getPieces(brd, new_r, new_c) != 0:
            #If count is odd then this square belongs to the opposing player
            if count % 2 == 1:
                if init_stack > getPieces(brd, new_r, new_c):
                    return True
    
# This function actually adjusts the board and the capture numbers
#  ... based on a capturing move that is known to be legal
def doCapture(state, r, c, direct):
    player = state['Turn']
    brd = state['Board']
    init_stack = getPieces(brd, r, c)

    if direct == 'N':
        d_row = -1
        d_col = 0
    elif direct == 'S':
        d_row = 1
        d_col = 0
    elif direct == 'E':
        d_row = 0
        d_col = 1
    elif direct == 'W':
        d_row = 0
        d_col = -1

    done = False
    new_r = r
    new_c = c
    count = init_stack
    while not done:
        new_r += d_row
        new_c += d_col

        # The capture move is done if you hit the edge of the board
        # ... or run out of pieces
        if (new_r < 0) or (new_r > 5):
            done = True
        elif (new_c < 0) or (new_c > 5):
            done = True
        elif count == 0:
            done = True

        # If the square belongs to you, you just drop a piece and move on
        elif color(new_r, new_c) == player:
            count = count - 1
            pieces = getPieces(brd, new_r, new_c)
            setPieces(brd, new_r, new_c, pieces + 1)
        else:
            count = count - 1
            pieces = getPieces(brd, new_r, new_c)
            
            #You can only capture a stack smaller than your starting stack
            #And you can't capture a size zero stack
            if (init_stack > pieces) and (pieces > 0):
                setPieces(brd, new_r, new_c, 0)
                if player == 'Dark':
                    state['DarkCapture'] += pieces + 1
                else:
                    state['LightCapture'] += pieces + 1
            #If you don't have a larger stack,
            #   or if your opponent has zero pieces
            #   then you just drop a piece
            #       ... and grow your opponent's stack
            else:
                setPieces(brd, new_r, new_c, pieces + 1)

    # When you hit the edge or run out of pieces
    # Then whatever you have left goes to the initial square
    setPieces(brd, r, c, count)

    return state

        
    
