## This is an custom player who always captures when she is able to capture and has an even amount of pieces.

import GameRules
import random

def name():
    return 'CustomPlayer1'

CARDINALS = {'N', 'E', 'S', 'W'}

def count_my_pieces(state):
    board = state['Board']
    me = state['Turn']
    total = 0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) == me:
                total += GameRules.getPieces(board, r, c)
    return total

def evaluate_after_move(state_after, player):
    opponent = state_after['Turn']
    base_capture = state_after['DarkCapture'] if opponent == 'Dark' else state_after['LightCapture']
    
    # Scan opponent replies to gauge the biggest capture we might give up
    worst_capture = 0
    for opp_mv in GameRules.getAllLegalMoves(state_after):
        if opp_mv['Direction'] in CARDINALS:
            result = GameRules.playMove(state_after, opp_mv)
            if result is None:
                continue
            if opponent == 'Dark':
                delta = result['DarkCapture'] - base_capture
            else:
                delta = result['LightCapture'] - base_capture
            if delta > worst_capture:
                worst_capture = delta

    future_state = GameRules.copyState(state_after)
    future_state['Turn'] = player
    mobility = len(GameRules.getAllLegalMoves(future_state))
    return worst_capture, mobility

def score_move(state, move, player):
    next_state = GameRules.playMove(state, move)
    if next_state is None:
        return (float('inf'), float('inf'), float('inf'))
    threat, mobility = evaluate_after_move(next_state, player)
    if player == 'Dark':
        immediate_gain = next_state['DarkCapture'] - state['DarkCapture']
    else:
        immediate_gain = next_state['LightCapture'] - state['LightCapture']
    return (threat, -immediate_gain, -mobility)

def getMove(state):
    legal_moves = GameRules.getAllLegalMoves(state)
    me = state['Turn']
    even_total = (count_my_pieces(state) % 2 == 0)

    capture_moves = [mv for mv in legal_moves if mv['Direction'] in CARDINALS]
    non_capture_moves = [mv for mv in legal_moves if mv['Direction'] not in CARDINALS]

    if even_total and capture_moves:
        candidate_moves = capture_moves
    elif non_capture_moves:
        candidate_moves = non_capture_moves
    else:
        candidate_moves = capture_moves  # must capture if no safe alternative

    best_score = (float('inf'), float('inf'), float('inf'))
    best_move = None
    for mv in candidate_moves:
        move_score = score_move(state, mv, me)
        if move_score < best_score:
            best_score = move_score
            best_move = mv

    return best_move if best_move is not None else legal_moves[0]