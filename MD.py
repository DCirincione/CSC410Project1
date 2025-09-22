# Minimax player with alpha–beta pruning

import math
import random
import GameRules

########### Public API ###########

def name():
    return "MD"

def getMove(state):
    """Choose a move using iterative deepening + alpha-beta (depth-limited)."""
    # Safety: if no moves exist, just return something benign (engine should not call us then)
    legal = GameRules.getAllLegalMoves(state)
    if not legal:
        return {"Row": 0, "Col": 0, "Direction": "NW"}  # dummy
    
    # Iterative deepening for better move ordering (fixed max depth to keep runtime reasonable)
    best_move = legal[0]
    max_depth = 3  # tweakable; increase if performance allows
    alpha = -math.inf
    beta = math.inf

    # Initial ordering: captures first
    ordered = order_moves(state, legal)

    for depth in range(1, max_depth + 1):
        value, move = alphabeta_root(state, ordered, depth, alpha, beta)
        if move is not None:
            best_move = move
        # Re-order using PV move first on next iteration
        ordered = move_to_front(ordered, best_move)

    return best_move

########### Minimax + Alpha-Beta ###########

def alphabeta_root(state, moves, depth, alpha, beta):
    """Root driver: maximizing appears if it's Light's perspective? We define maximizing as 'player to move'."""
    best_val = -math.inf
    best_move = None

    # Root uses player-to-move as "maximizing"
    for mv in moves:
        child = GameRules.playMove(state, mv)
        if child is None:  # illegal, skip
            continue
        val = alphabeta(child, depth - 1, alpha, beta, maximizing=False, root_player=state["Turn"])
        if val > best_val:
            best_val = val
            best_move = mv
        alpha = max(alpha, best_val)
        if alpha >= beta:
            break  # prune
    return best_val, best_move

def alphabeta(state, depth, alpha, beta, maximizing, root_player):
    # Terminal or depth cutoff
    if GameRules.isGameOver(state):
        # Apply end-game bonus to evaluate true final score from this state
        final_state = GameRules.endGame(state)
        return terminal_evaluation(final_state, root_player)

    if depth == 0:
        return evaluate(state, root_player)

    legal = GameRules.getAllLegalMoves(state)
    if not legal:
        # If no moves but GameRules.isGameOver returned False (shouldn't happen),
        # treat as terminal with endGame() for safety.
        final_state = GameRules.endGame(state)
        return terminal_evaluation(final_state, root_player)

    # Move ordering: captures first, then by heuristic
    ordered = order_moves(state, legal)

    if maximizing:
        value = -math.inf
        for mv in ordered:
            child = GameRules.playMove(state, mv)
            if child is None:
                continue
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False, root_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # prune
        return value
    else:
        value = math.inf
        for mv in ordered:
            child = GameRules.playMove(state, mv)
            if child is None:
                continue
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True, root_player))
            beta = max(-math.inf, min(beta, value))
            if alpha >= beta:
                break  # prune
        return value

########### Evaluation Functions ###########

def terminal_evaluation(state, root_player):
    """Exact score at game end from root_player perspective.
    Positive is good for root_player, negative favors opponent.
    """
    # At end, the capture counts include the largest-stack bonus already.
    light_score = state.get("LightCapture", 0)
    dark_score = state.get("DarkCapture", 0)

    if root_player == "Light":
        diff = light_score - dark_score
    else:
        diff = dark_score - light_score

    # Tie-breaker: Light wins ties. Convert that to +/- epsilon.
    if diff == 0:
        if root_player == "Light":
            return 0.001  # tiny edge for Light in ties
        else:
            return -0.001
    return diff

def evaluate(state, root_player):
    """Heuristic evaluation at cutoff depth.
    Mix material, mobility, max-stack potential, and opponent mobility pressure.
    """
    light_cap = state.get("LightCapture", 0)
    dark_cap = state.get("DarkCapture", 0)
    brd = state.get("Board", [0]*36)

    # Material (captures so far)
    if root_player == "Light":
        material = light_cap - dark_cap
    else:
        material = dark_cap - light_cap

    # Largest stack differential (proxy for end-game bonus potential)
    light_max, dark_max = max_stacks(brd)
    if root_player == "Light":
        maxstack_term = light_max - dark_max
    else:
        maxstack_term = dark_max - light_max

    # Mobility (favor more legal moves; captures are more valuable)
    my_moves = GameRules.getAllLegalMoves(state)
    my_cap_count = sum(1 for m in my_moves if m["Direction"] in ("N","E","S","W"))
    my_move_score = 0.25*len(my_moves) + 0.75*my_cap_count

    # Approximate opponent mobility by flipping turn (simulate turn pass by applying a null?)
    # We can't pass a turn legally; instead, estimate by toggling 'Turn' and counting moves on a shallow copy.
    opp_state = copy_turn_only(state)
    opp_state["Turn"] = "Dark" if state["Turn"] == "Light" else "Light"
    opp_moves = GameRules.getAllLegalMoves(opp_state)
    opp_cap_count = sum(1 for m in opp_moves if m["Direction"] in ("N","E","S","W"))
    opp_move_score = 0.25*len(opp_moves) + 0.75*opp_cap_count

    mobility = my_move_score - opp_move_score

    # Weighted sum
    # Weights tuned conservatively to avoid overfitting
    score = (2.0 * material) + (0.6 * maxstack_term) + (0.5 * mobility)

    # Tiny nudge for Light tie rule when material == 0 and stacks equal
    if score == 0 and root_player == "Light":
        score += 0.001

    return score

def max_stacks(board_list):
    """Return (light_max, dark_max) from the flat 6x6 board representation.
    Square (0,0) is Dark; colors alternate checkerboard-style.
    """
    light_max = 0
    dark_max = 0
    for idx, cnt in enumerate(board_list):
        r = idx // 6
        c = idx % 6
        if ((r + c) % 2) == 0:  # Dark squares at (0,0) even parity per GameRules comment
            dark_max = max(dark_max, cnt)
        else:
            light_max = max(light_max, cnt)
    return light_max, dark_max

def copy_turn_only(state):
    """Shallow copy the state keys sufficient for counting moves with a flipped turn."""
    new_state = {
        "Turn": state["Turn"],
        "LightCapture": state.get("LightCapture", 0),
        "DarkCapture": state.get("DarkCapture", 0),
        "Board": list(state.get("Board", [])),
    }
    return new_state

########### Move Ordering ###########

def order_moves(state, moves):
    """Sort moves to improve alpha-beta pruning.
    Priority:
      1) Captures first (cardinal directions)
      2) Moves yielding larger immediate material gain
      3) Otherwise prefer moves that increase our max stack size
    """
    # Pre-evaluate children cheaply to guide ordering
    scored = []
    for mv in moves:
        is_cap = 1 if mv["Direction"] in ("N","E","S","W") else 0
        child = GameRules.playMove(state, mv)
        if child is None:
            gain = -999
            stack_gain = -999
        else:
            if state["Turn"] == "Light":
                gain = child.get("LightCapture", 0) - state.get("LightCapture", 0)
            else:
                gain = child.get("DarkCapture", 0) - state.get("DarkCapture", 0)
            # Stack potential heuristic: change in own max stack
            before_l, before_d = max_stacks(state["Board"])
            after_l, after_d = max_stacks(child["Board"])
            if state["Turn"] == "Light":
                stack_gain = (after_l - before_l) - (after_d - before_d)
            else:
                stack_gain = (after_d - before_d) - (after_l - before_l)
        scored.append((is_cap, gain, stack_gain, mv))

    # Sort descending by (is_cap, gain, stack_gain)
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [t[3] for t in scored]

def move_to_front(moves, best):
    """Place best move at the front of move list for next iteration (PV ordering)."""
    if best not in moves:
        return moves
    out = [best] + [m for m in moves if m is not best]
    return out
