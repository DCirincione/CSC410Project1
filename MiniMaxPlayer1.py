'''
CSC410 AI player featuring alpha-beta minimax search with adaptive depth extensions,
mobility forecasting, and a weighted evaluation that blends capture margins, remaining material, mobility options, centre/stack leverage, edge risk, and liberty counts. Move ordering prioritises captures and strong stacks, while added heuristics mitigate stalemates; inline commentary explains each tuning knob for instructional clarity.
'''

#imports
import math
import GameRules


CARDINALS = {'N', 'E', 'S', 'W'}
#if default depth before any extensions kick in
BASE_DEPTH = 3
#if cap on how many bonus plies adaptive search may add
MAX_EXTRA_DEPTH = 3
#if terminal scorer treats any capture lead beyond this as decisive
WIN_SCORE = 1_000_000

#if core capture/material weights that are tweaked often
CAPTURE_WEIGHT = 16.0             #if capture differential is the primary score driver
PIECE_WEIGHT = 1.35               #if remaining material should still influence value
MOBILITY_WEIGHT = 0.6             #if raw legal move count provides baseline flexibility
CAPTURE_MOVE_WEIGHT = 0.8         #if availability of immediate captures adds pressure
CENTER_WEIGHT = 0.35              #if keeping stacks near the centre earns credit
STACK_WEIGHT = 0.45               #if taller friendly stacks indicate initiative
LIBERTY_WEIGHT = 1.4              #if more active origins reduce stalemate risk

#if structural bonuses and penalties for board shape
EDGE_PENALTY_WEIGHT = 0.35        #if losing edge tug-of-war should hurt overall score
FROZEN_WEIGHT = 3.0               #if immobile stacks represent significant liability
EDGE_OVERLOAD_WEIGHT = 1.8        #if high rim occupancy must be punished aggressively
MEGA_STACK_PENALTY = 4.0          #if giant towers tend to self-trap unless broken up
ORIGIN_REWARD = 2.0               #if maintaining several distinct move origins is valuable
SPLIT_INCENTIVE = 1.5             #if in danger, nudge search toward splitting big stacks

#if mobility safety thresholds for deeper search and evaluation
LOW_MOBILITY_THRESHOLD = 4        #if four or fewer moves signals a fragile position
LOW_MOBILITY_WEIGHT = 4.5         #if small mobility deficits should already cost points
MOBILITY_COLLAPSE_WEIGHT = 2.8    #if collapsing mobility deserves a quadratic penalty
FORECAST_PENALTY_WEIGHT = 6.0     #if lookahead warnings about low mobility must be heeded
FORECAST_BONUS_WEIGHT = 1.8       #if forecasted breathing room earns a modest boost
FUTURE_ZERO_PENALTY = 60.0        #if predicted stalemate next ply is nearly catastrophic


def name():
    return 'MiniMaxPlayer1'


def getMove(state):
    #if query all legal actions for current board
    legal_moves = GameRules.getAllLegalMoves(state)
    if not legal_moves:
        #if engine provides no options, surrender turn gracefully
        return None

    player = state['Turn']
    #if fewer moves, search deeper to see freeze traps
    depth = choose_depth(state, len(legal_moves))

    children = []
    #if collect playable child states before running deeper search
    for move in legal_moves:
        next_state = GameRules.playMove(state, move)
        if next_state is None:
            #if illegal move surfaces, just skip it
            continue
        children.append((move, next_state))

    if not children:
        #if every move failed during simulation, fall back to first legal move
        return legal_moves[0]

    #if capture-first ordering helps pruning later on
    children = order_children(state, children, player)

    best_move = children[0][0]
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf

    for move, next_state in children:
        #if evaluate child position via alpha-beta recursion
        value = alpha_beta(next_state, depth - 1, alpha, beta, player)
        if value > best_value:
            best_value = value
            best_move = move
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break

    #if final decision for this ply
    return best_move


def choose_depth(state, legal_moves_count=None):
    #if tune search depth dynamically from remaining pieces and mobility
    if legal_moves_count is None:
        legal_moves_count = len(GameRules.getAllLegalMoves(state))

    pieces_remaining = sum(state['Board'])
    extra = 0
    if pieces_remaining <= 22:
        #if late game, allow more lookahead automatically
        extra += 1
    if pieces_remaining <= 14:
        #if endgame, favour deeper view of forced lines
        extra += 1

    if legal_moves_count <= 6:
        #if mobility tightens, add emergency depth
        extra += 1
    if legal_moves_count <= 3:
        #if mobility is extremely low, push depth again
        extra += 1
    if legal_moves_count <= 2:
        extra += 2
        player = state['Turn']
        opponent = other_player(player)
        capture_edge = capture_total(state, player) - capture_total(state, opponent)
        if capture_edge >= 2:
            #if we're ahead on captures, invest one more ply to avoid stalemate blunders
            extra += 1

    extra = min(extra, MAX_EXTRA_DEPTH)
    #if clamp depth to valid range and ensure at least one ply
    return max(1, BASE_DEPTH + extra)


def alpha_beta(state, depth, alpha, beta, player):
    #if classic alpha-beta minimax with move ordering
    legal_moves = GameRules.getAllLegalMoves(state)
    if depth == 0 or not legal_moves:
        #if reached leaf or no moves, rely on evaluation function
        return evaluate_state(state, player, legal_moves)

    maximizing = state['Turn'] == player
    #if sort moves so pruning hits likely captures/stacks first
    ordered = order_moves_simple(state, legal_moves)

    if maximizing:
        #if we are maximizing player, initialise value low
        value = -math.inf
        for move in ordered:
            next_state = GameRules.playMove(state, move)
            if next_state is None:
                continue
            #if recurse with decreased depth to evaluate child
            value = max(value, alpha_beta(next_state, depth - 1, alpha, beta, player))
            alpha = max(alpha, value)
            if alpha >= beta:
                #if pruning branch because opponent already has better option
                break
        if value == -math.inf:
            #if no legal children survived, fall back to static eval
            return evaluate_state(state, player, legal_moves)
        return value

    #if minimizing branch mirrors the maximizing case
    value = math.inf
    for move in ordered:
        next_state = GameRules.playMove(state, move)
        if next_state is None:
            continue
        #if recurse assuming opponent tries to minimize our value
        value = min(value, alpha_beta(next_state, depth - 1, alpha, beta, player))
        beta = min(beta, value)
        if beta <= alpha:
            #if prune because maximizing side already secured better line
            break
    if value == math.inf:
        #if no valid replies, evaluate statically
        return evaluate_state(state, player, legal_moves)
    return value


def evaluate_state(state, player, legal_moves=None):
    opponent = other_player(player)

    if legal_moves is None:
        #if caller already fetched moves, reuse them to save work
        legal_moves = GameRules.getAllLegalMoves(state)

    if not legal_moves:
        #if no moves, endGame handles bonus capture and winner
        final_state = GameRules.endGame(state)
        return terminal_score(final_state, player)

    #if tally capture race to know scoreboard advantage
    my_capture = capture_total(state, player)
    opp_capture = capture_total(state, opponent)
    capture_diff = my_capture - opp_capture

    #if material counts help balance greedy capture play
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
    #if opponent metrics to compare against our mobility profile
    (
        opp_moves,
        opp_capture_moves,
        opp_frozen_weight,
        opp_edge_weight,
        opp_liberties,
        opp_piece_total,
        opp_max_stack,
    ) = mobility_profile(state, opponent)

    #if compare our activity and structural stats versus opponent
    mobility_diff = my_moves - opp_moves
    capture_option_diff = my_capture_moves - opp_capture_moves
    frozen_diff = opp_frozen_weight - my_frozen_weight
    edge_diff = opp_edge_weight - my_edge_weight
    liberty_diff = my_liberties - opp_liberties
    edge_ratio = (my_edge_weight / my_piece_total) if my_piece_total else 0.0  #if share of our pieces hugging edge
    opp_edge_ratio = (opp_edge_weight / opp_piece_total) if opp_piece_total else 0.0  #if opponent edge share

    #if center and stack contrast reflect positional grip
    center_diff = central_control(state, player) - central_control(state, opponent)
    stack_diff = max_stack_height(state, player) - max_stack_height(state, opponent)

    mobility_pressure = 0.0
    if my_moves < LOW_MOBILITY_THRESHOLD:
        #if we are close to stalemating ourselves, punish harshly
        mobility_pressure -= LOW_MOBILITY_WEIGHT * (LOW_MOBILITY_THRESHOLD - my_moves)
        mobility_pressure -= MOBILITY_COLLAPSE_WEIGHT * (LOW_MOBILITY_THRESHOLD - my_moves) ** 2
    if opp_moves < LOW_MOBILITY_THRESHOLD:
        #if opponent is locked, small bonus for keeping pressure
        mobility_pressure += (LOW_MOBILITY_WEIGHT * 0.7) * (LOW_MOBILITY_THRESHOLD - opp_moves)

    #if blend our core heuristics: capture, material, activity, positioning
    score = (
        CAPTURE_WEIGHT * capture_diff          #if prioritize current capture lead
        + PIECE_WEIGHT * piece_diff            #if reward extra material on board
        + MOBILITY_WEIGHT * mobility_diff      #if prefer having more legal moves
        + CAPTURE_MOVE_WEIGHT * capture_option_diff  #if value pressure from available captures
        + CENTER_WEIGHT * center_diff          #if interior control matters
        + STACK_WEIGHT * stack_diff            #if taller friendly stacks threaten tempo
        + LIBERTY_WEIGHT * liberty_diff        #if more active origins keep flexibility
    )

    score += mobility_pressure                      #if dynamic mobility penalties/bonuses
    score += FROZEN_WEIGHT * frozen_diff             #if punish own frozen stacks versus opponent
    score += EDGE_PENALTY_WEIGHT * edge_diff         #if offset edge exposure differential
    #if threshold forces us to reduce over-reliance on rim stacks
    score -= EDGE_OVERLOAD_WEIGHT * max(0.0, edge_ratio - 0.55)
    #if opponent clings to edge, encourage us to keep them there
    score += EDGE_OVERLOAD_WEIGHT * max(0.0, opp_edge_ratio - 0.55)
    if my_max_stack >= 5:
        #if we built a mega stack, urge breaking it up
        score -= MEGA_STACK_PENALTY * (my_max_stack - 4)
    if opp_max_stack >= 5:
        #if opponent towers, reward chances to attack it
        score += (MEGA_STACK_PENALTY * 0.7) * (opp_max_stack - 4)
    if my_liberties >= 3:
        #if we maintain multiple active origins, boost score
        score += ORIGIN_REWARD * (my_liberties - 2)
    if opp_liberties >= 3:
        #if opponent also spreads out, dampen the benefit
        score -= (ORIGIN_REWARD * 0.6) * (opp_liberties - 2)

    if legal_moves and (
        my_moves <= LOW_MOBILITY_THRESHOLD
        or opp_moves <= LOW_MOBILITY_THRESHOLD
        or len(legal_moves) <= LOW_MOBILITY_THRESHOLD + 2
    ):
        #if we're near a freeze, peek one ply deeper to judge risk
        worst_future, best_future = forecast_mobility_swing(state, player, legal_moves)
        if worst_future is not None:
            if worst_future == 0:
            #if forecast says zero moves, treat it as near certain loss
                score -= FUTURE_ZERO_PENALTY
            elif worst_future < LOW_MOBILITY_THRESHOLD:
                score -= FORECAST_PENALTY_WEIGHT * (LOW_MOBILITY_THRESHOLD - worst_future)
        if best_future is not None and best_future > LOW_MOBILITY_THRESHOLD:
            headroom = min(best_future, LOW_MOBILITY_THRESHOLD * 2) - LOW_MOBILITY_THRESHOLD
            score += FORECAST_BONUS_WEIGHT * headroom
        if worst_future is not None and worst_future <= 2 and my_max_stack >= 5:
            #if future is shaky and we have a mega stack, encourage a split move
            score += SPLIT_INCENTIVE * (my_max_stack - 4)

    #if aggregated heuristic value for current player perspective
    return score


def terminal_score(state, player):
    #if derive final winner purely from capture totals
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
        #if prefer capture moves, bigger gains, and larger stacks first
        return (-is_capture, -immediate_gain, -stack_size)

    #if we search best looking moves first, alpha-beta cuts more branches
    return sorted(children, key=priority)


def order_moves_simple(state, moves):
    def priority(move):
        is_capture = 1 if move['Direction'] in CARDINALS else 0
        stack_size = GameRules.getPieces(state['Board'], move['Row'], move['Col'])
        #if rank by capture potential then larger originating stacks
        return (-is_capture, -stack_size)

    #if helper shares ordering logic for other callers
    return sorted(moves, key=priority)


def mobility_profile(state, player):
    #if summarise mobility, edge load, and stack risks for a player
    probe = GameRules.copyState(state)
    probe['Turn'] = player
    moves = GameRules.getAllLegalMoves(probe)
    capture_moves = 0
    active_origins = set()
    for mv in moves:
        if mv['Direction'] in CARDINALS:
            capture_moves += 1
        #if record squares that produced moves so we know active origins
        active_origins.add((mv['Row'], mv['Col']))

    #if evaluate board features for the player copy
    board = probe['Board']
    frozen_weight = 0      #if total pieces sitting on origins without moves
    edge_weight = 0        #if cumulative stack size on rim squares
    piece_total = 0        #if total material counted for ratio calculations
    max_stack = 0          #if tallest stack we encounter while scanning
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) != player:
                continue
            pieces = GameRules.getPieces(board, r, c)
            if pieces == 0:
                continue
            piece_total += pieces       #if accumulate material for later ratios
            if pieces > max_stack:
                max_stack = pieces      #if maintain maximum stack height seen so far
            if r in (0, 5) or c in (0, 5):
                #if accumulate rim exposure to penalize edge-heavy distributions
                edge_weight += pieces
            if (r, c) not in active_origins:
                #if a stack cannot move this turn, treat those pieces as frozen risk
                frozen_weight += pieces

    #if return tuple: move count, captures, frozen load, edge load, origins, total pieces, tallest stack
    return (
        len(moves),
        capture_moves,
        frozen_weight,
        edge_weight,
        len(active_origins),
        piece_total,
        max_stack,
    )


#if forecast sampling limits stay near this helper
FORECAST_MOVE_LIMIT = 8
FORECAST_REPLY_LIMIT = 6


def forecast_mobility_swing(state, player, legal_moves):
    if not legal_moves:
        #if no legal moves to explore, nothing to forecast
        return None, None

    worst_future = math.inf
    best_future = -math.inf

    if state['Turn'] == player:
        #if simulate our best-looking moves first for faster pruning
        ordered_moves = order_moves_simple(state, legal_moves)
        for move in ordered_moves[:FORECAST_MOVE_LIMIT]:
            next_state = GameRules.playMove(state, move)
            if next_state is None:
                continue
            opp_moves = GameRules.getAllLegalMoves(next_state)
            if not opp_moves:
                #if opponent cannot move after our choice, we effectively stay safe
                mobility_value = LOW_MOBILITY_THRESHOLD * 2
                worst_future = min(worst_future, mobility_value)
                best_future = max(best_future, mobility_value)
                continue

            min_future = math.inf
            max_future = -math.inf
            #if opponent replies are likewise sorted for relevance
            ordered_replies = order_moves_simple(next_state, opp_moves)
            for opp_move in ordered_replies[:FORECAST_REPLY_LIMIT]:
                future_state = GameRules.playMove(next_state, opp_move)
                if future_state is None:
                    continue
                mobility, *_ = mobility_profile(future_state, player)
                min_future = min(min_future, mobility)
                max_future = max(max_future, mobility)

            if min_future == math.inf:
                #if opponent had no valid replies, treat worst case as zero mobility
                min_future = 0
            if max_future == -math.inf:
                #if likewise no replies, best case defaults to zero as well
                max_future = 0
            worst_future = min(worst_future, min_future)
            best_future = max(best_future, max_future)
    else:
        #if current turn belongs to opponent, sample their best replies directly
        ordered_replies = order_moves_simple(state, legal_moves)
        for opp_move in ordered_replies[:FORECAST_REPLY_LIMIT]:
            next_state = GameRules.playMove(state, opp_move)
            if next_state is None:
                continue
            mobility, *_ = mobility_profile(next_state, player)
            worst_future = min(worst_future, mobility)
            best_future = max(best_future, mobility)

    if worst_future == math.inf:
        #if no values recorded, treat forecast as unavailable
        worst_future = None
    if best_future == -math.inf:
        #if best case never updated, mark as unknown
        best_future = None

    return worst_future, best_future


def central_control(state, player):
    #if favour pieces nearer the centre weighted by stack size
    board = state['Board']
    score = 0.0
    #if iterate every square to accumulate centre influence
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) != player:
                continue
            pieces = GameRules.getPieces(board, r, c)
            if pieces == 0:
                continue
            #if compute Manhattan distance from centre (2.5, 2.5)
            dist = abs(2.5 - r) + abs(2.5 - c)
            #if closer to board centre yields larger bonus (3 - distance)
            score += pieces * (3.0 - dist)
    return score


def max_stack_height(state, player):
    #if track maximum stack height for tempo/combat potential
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
    #if sum all stack sizes on player's colour squares
    board = state['Board']
    total = 0
    for r in range(6):
        for c in range(6):
            if GameRules.color(r, c) == player:
                total += GameRules.getPieces(board, r, c)
    return total


def capture_total(state, player):
    #if capture counts stored separately for each colour
    if player == 'Dark':
        return state['DarkCapture']
    return state['LightCapture']


def other_player(player):
    return 'Light' if player == 'Dark' else 'Dark'
