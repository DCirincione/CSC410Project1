"""MiniMax-based player using alpha-beta pruning and a composite heuristic."""

import math
import GameRules


CARDINALS = {'N', 'E', 'S', 'W'}
BASE_DEPTH = 3
MAX_EXTRA_DEPTH = 3
WIN_SCORE = 1_000_000
CAPTURE_WEIGHT = 16.0
PIECE_WEIGHT = 1.35
MOBILITY_WEIGHT = 0.6
CAPTURE_MOVE_WEIGHT = 0.8
CENTER_WEIGHT = 0.35
STACK_WEIGHT = 0.45
EDGE_PENALTY_WEIGHT = 0.35
FROZEN_WEIGHT = 3.0
EDGE_OVERLOAD_WEIGHT = 1.8
MEGA_STACK_PENALTY = 4.0
ORIGIN_REWARD = 2.0
SPLIT_INCENTIVE = 1.5
LOW_MOBILITY_THRESHOLD = 4
LOW_MOBILITY_WEIGHT = 4.5
FORECAST_PENALTY_WEIGHT = 6.0
FORECAST_BONUS_WEIGHT = 1.8
FORECAST_MOVE_LIMIT = 8
FORECAST_REPLY_LIMIT = 6
LIBERTY_WEIGHT = 1.4
MOBILITY_COLLAPSE_WEIGHT = 2.8
FUTURE_ZERO_PENALTY = 60.0


def name():
    return 'MiniMaxPlayer1'


def getMove(state):
    legal_moves = GameRules.getAllLegalMoves(state)
    if not legal_moves:
        return None

    player = state['Turn']
    depth = choose_depth(state, len(legal_moves))

    children = []
    for move in legal_moves:
        next_state = GameRules.playMove(state, move)
        if next_state is None:
            continue
        children.append((move, next_state))

    if not children:
        return legal_moves[0]

    children = order_children(state, children, player)

    best_move = children[0][0]
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf

    for move, next_state in children:
        value = alpha_beta(next_state, depth - 1, alpha, beta, player)
        if value > best_value:
            best_value = value
            best_move = move
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break

    return best_move


def choose_depth(state, legal_moves_count=None):
    if legal_moves_count is None:
        legal_moves_count = len(GameRules.getAllLegalMoves(state))

    pieces_remaining = sum(state['Board'])
    extra = 0
    if pieces_remaining <= 22:
        extra += 1
    if pieces_remaining <= 14:
        extra += 1

    if legal_moves_count <= 6:
        extra += 1
    if legal_moves_count <= 3:
        extra += 1
    if legal_moves_count <= 2:
        extra += 2
        player = state['Turn']
        opponent = other_player(player)
        capture_edge = capture_total(state, player) - capture_total(state, opponent)
        if capture_edge >= 2:
            extra += 1

    extra = min(extra, MAX_EXTRA_DEPTH)
    return max(1, BASE_DEPTH + extra)


def alpha_beta(state, depth, alpha, beta, player):
    legal_moves = GameRules.getAllLegalMoves(state)
    if depth == 0 or not legal_moves:
        return evaluate_state(state, player, legal_moves)

    maximizing = state['Turn'] == player
    ordered = order_moves_simple(state, legal_moves)

    if maximizing:
        value = -math.inf
        for move in ordered:
            next_state = GameRules.playMove(state, move)
            if next_state is None:
                continue
            value = max(value, alpha_beta(next_state, depth - 1, alpha, beta, player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if value == -math.inf:
            return evaluate_state(state, player, legal_moves)
        return value

    value = math.inf
    for move in ordered:
        next_state = GameRules.playMove(state, move)
        if next_state is None:
            continue
        value = min(value, alpha_beta(next_state, depth - 1, alpha, beta, player))
        beta = min(beta, value)
        if beta <= alpha:
            break
    if value == math.inf:
        return evaluate_state(state, player, legal_moves)
    return value


def evaluate_state(state, player, legal_moves=None):
    opponent = other_player(player)

    if legal_moves is None:
        legal_moves = GameRules.getAllLegalMoves(state)

    if not legal_moves:
        final_state = GameRules.endGame(state)
        return terminal_score(final_state, player)

    my_capture = capture_total(state, player)
    opp_capture = capture_total(state, opponent)
    capture_diff = my_capture - opp_capture

    my_pieces = count_pieces(state, player)
    opp_pieces = count_pieces(state, opponent)
    piece_diff = my_pieces - opp_pieces

    (
        my_moves,
        my_capture_moves,
        my_frozen_weight,
        my_edge_weight,
        my_liberties,
        my_piece_total,
        my_max_stack,
    ) = mobility_profile(state, player)
    (
        opp_moves,
        opp_capture_moves,
        opp_frozen_weight,
        opp_edge_weight,
        opp_liberties,
        opp_piece_total,
        opp_max_stack,
    ) = mobility_profile(state, opponent)

    mobility_diff = my_moves - opp_moves
    capture_option_diff = my_capture_moves - opp_capture_moves
    frozen_diff = opp_frozen_weight - my_frozen_weight
    edge_diff = opp_edge_weight - my_edge_weight
    liberty_diff = my_liberties - opp_liberties
    edge_ratio = (my_edge_weight / my_piece_total) if my_piece_total else 0.0
    opp_edge_ratio = (opp_edge_weight / opp_piece_total) if opp_piece_total else 0.0

    center_diff = central_control(state, player) - central_control(state, opponent)
    stack_diff = max_stack_height(state, player) - max_stack_height(state, opponent)

    mobility_pressure = 0.0
    if my_moves < LOW_MOBILITY_THRESHOLD:
        mobility_pressure -= LOW_MOBILITY_WEIGHT * (LOW_MOBILITY_THRESHOLD - my_moves)
        mobility_pressure -= MOBILITY_COLLAPSE_WEIGHT * (LOW_MOBILITY_THRESHOLD - my_moves) ** 2
    if opp_moves < LOW_MOBILITY_THRESHOLD:
        mobility_pressure += (LOW_MOBILITY_WEIGHT * 0.7) * (LOW_MOBILITY_THRESHOLD - opp_moves)

    score = (
        CAPTURE_WEIGHT * capture_diff
        + PIECE_WEIGHT * piece_diff
        + MOBILITY_WEIGHT * mobility_diff
        + CAPTURE_MOVE_WEIGHT * capture_option_diff
        + CENTER_WEIGHT * center_diff
        + STACK_WEIGHT * stack_diff
        + LIBERTY_WEIGHT * liberty_diff
    )

    score += mobility_pressure
    score += FROZEN_WEIGHT * frozen_diff
    score += EDGE_PENALTY_WEIGHT * edge_diff
    score -= EDGE_OVERLOAD_WEIGHT * max(0.0, edge_ratio - 0.55)
    score += EDGE_OVERLOAD_WEIGHT * max(0.0, opp_edge_ratio - 0.55)
    if my_max_stack >= 5:
        score -= MEGA_STACK_PENALTY * (my_max_stack - 4)
    if opp_max_stack >= 5:
        score += (MEGA_STACK_PENALTY * 0.7) * (opp_max_stack - 4)
    if my_liberties >= 3:
        score += ORIGIN_REWARD * (my_liberties - 2)
    if opp_liberties >= 3:
        score -= (ORIGIN_REWARD * 0.6) * (opp_liberties - 2)

    if legal_moves and (
        my_moves <= LOW_MOBILITY_THRESHOLD
        or opp_moves <= LOW_MOBILITY_THRESHOLD
        or len(legal_moves) <= LOW_MOBILITY_THRESHOLD + 2
    ):
        worst_future, best_future = forecast_mobility_swing(state, player, legal_moves)
        if worst_future is not None:
            if worst_future == 0:
                score -= FUTURE_ZERO_PENALTY
            elif worst_future < LOW_MOBILITY_THRESHOLD:
                score -= FORECAST_PENALTY_WEIGHT * (LOW_MOBILITY_THRESHOLD - worst_future)
        if best_future is not None and best_future > LOW_MOBILITY_THRESHOLD:
            headroom = min(best_future, LOW_MOBILITY_THRESHOLD * 2) - LOW_MOBILITY_THRESHOLD
            score += FORECAST_BONUS_WEIGHT * headroom
        if worst_future is not None and worst_future <= 2 and my_max_stack >= 5:
            score += SPLIT_INCENTIVE * (my_max_stack - 4)

    return score


def terminal_score(state, player):
    my_capture = capture_total(state, player)
    opp_capture = capture_total(state, other_player(player))
    diff = my_capture - opp_capture
    if diff > 0:
        return WIN_SCORE
    if diff < 0:
        return -WIN_SCORE
    return 0.0


def order_children(state, children, player):
    def priority(item):
        move, next_state = item
        is_capture = 1 if move['Direction'] in CARDINALS else 0
        immediate_gain = capture_total(next_state, player) - capture_total(state, player)
        stack_size = GameRules.getPieces(state['Board'], move['Row'], move['Col'])
        return (-is_capture, -immediate_gain, -stack_size)

    return sorted(children, key=priority)


def order_moves_simple(state, moves):
    def priority(move):
        is_capture = 1 if move['Direction'] in CARDINALS else 0
        stack_size = GameRules.getPieces(state['Board'], move['Row'], move['Col'])
        return (-is_capture, -stack_size)

    return sorted(moves, key=priority)


def mobility_profile(state, player):
    probe = GameRules.copyState(state)
    probe['Turn'] = player
    moves = GameRules.getAllLegalMoves(probe)
    capture_moves = 0
    active_origins = set()
    for mv in moves:
        if mv['Direction'] in CARDINALS:
            capture_moves += 1
        active_origins.add((mv['Row'], mv['Col']))

    board = probe['Board']
    frozen_weight = 0
    edge_weight = 0
    piece_total = 0
    max_stack = 0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) != player:
                continue
            pieces = GameRules.getPieces(board, r, c)
            if pieces == 0:
                continue
            piece_total += pieces
            if pieces > max_stack:
                max_stack = pieces
            if r in (0, 5) or c in (0, 5):
                edge_weight += pieces
            if (r, c) not in active_origins:
                frozen_weight += pieces

    return (
        len(moves),
        capture_moves,
        frozen_weight,
        edge_weight,
        len(active_origins),
        piece_total,
        max_stack,
    )


def forecast_mobility_swing(state, player, legal_moves):
    if not legal_moves:
        return None, None

    worst_future = math.inf
    best_future = -math.inf

    if state['Turn'] == player:
        ordered_moves = order_moves_simple(state, legal_moves)
        for move in ordered_moves[:FORECAST_MOVE_LIMIT]:
            next_state = GameRules.playMove(state, move)
            if next_state is None:
                continue
            opp_moves = GameRules.getAllLegalMoves(next_state)
            if not opp_moves:
                mobility_value = LOW_MOBILITY_THRESHOLD * 2
                worst_future = min(worst_future, mobility_value)
                best_future = max(best_future, mobility_value)
                continue

            min_future = math.inf
            max_future = -math.inf
            ordered_replies = order_moves_simple(next_state, opp_moves)
            for opp_move in ordered_replies[:FORECAST_REPLY_LIMIT]:
                future_state = GameRules.playMove(next_state, opp_move)
                if future_state is None:
                    continue
                mobility, *_ = mobility_profile(future_state, player)
                min_future = min(min_future, mobility)
                max_future = max(max_future, mobility)

            if min_future == math.inf:
                min_future = 0
            if max_future == -math.inf:
                max_future = 0
            worst_future = min(worst_future, min_future)
            best_future = max(best_future, max_future)
    else:
        ordered_replies = order_moves_simple(state, legal_moves)
        for opp_move in ordered_replies[:FORECAST_REPLY_LIMIT]:
            next_state = GameRules.playMove(state, opp_move)
            if next_state is None:
                continue
            mobility, *_ = mobility_profile(next_state, player)
            worst_future = min(worst_future, mobility)
            best_future = max(best_future, mobility)

    if worst_future == math.inf:
        worst_future = None
    if best_future == -math.inf:
        best_future = None

    return worst_future, best_future


def central_control(state, player):
    board = state['Board']
    score = 0.0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) != player:
                continue
            pieces = GameRules.getPieces(board, r, c)
            if pieces == 0:
                continue
            dist = abs(2.5 - r) + abs(2.5 - c)
            score += pieces * (3.0 - dist)
    return score


def max_stack_height(state, player):
    board = state['Board']
    best = 0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) != player:
                continue
            pieces = GameRules.getPieces(board, r, c)
            if pieces > best:
                best = pieces
    return best


def count_pieces(state, player):
    board = state['Board']
    total = 0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) == player:
                total += GameRules.getPieces(board, r, c)
    return total


def capture_total(state, player):
    if player == 'Dark':
        return state['DarkCapture']
    return state['LightCapture']


def other_player(player):
    return 'Light' if player == 'Dark' else 'Dark'
