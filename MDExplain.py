# Drop-in player with Minimax + Alpha-Beta that writes a short summary per move.

import os
import math
import time
import GameRules

MAX_DEPTH = int(os.environ.get("MD_MAX_DEPTH", "3"))
TOPK = int(os.environ.get("MD_TOPK", "3"))
SUMMARY_FILE = os.environ.get("MD_SUMMARY_FILE", "MD_summary.log")
PRINT = os.environ.get("MD_PRINT", "0") in ("1","true","True")
SHOW_CANDIDATES = os.environ.get("MD_SHOW_CANDIDATES", "1") in ("1","true","True")

def name():
    return "MD"

############ I/O helpers ############

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _writeln(line):
    try:
        with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if PRINT:
        print(line, flush=True)

def _sep():
    _writeln("="*68)

def _fmt_move(mv):
    return f"({mv.get('Row')},{mv.get('Col')})-{mv.get('Direction')}"

############ Public API ############

def getMove(state):
    legal = GameRules.getAllLegalMoves(state)
    if not legal:
        return {"Row":0,"Col":0,"Direction":"NW"}

    root_player = state["Turn"]
    ordered = order_moves(state, legal)

    # Evaluate each root child fully to depth with alpha-beta; collect concise features.
    results = []
    alpha, beta = -math.inf, math.inf
    for mv in ordered:
        child = GameRules.playMove(state, mv)
        if child is None:
            continue
        val = alphabeta(child, MAX_DEPTH - 1, alpha, beta, maximizing=False, root_player=root_player)
        alpha = max(alpha, val)

        # Features for rationale
        if root_player == "Light":
            mat_gain = child.get("LightCapture",0) - state.get("LightCapture",0)
        else:
            mat_gain = child.get("DarkCapture",0) - state.get("DarkCapture",0)

        bl = state.get("Board", [0]*36)
        cl = child.get("Board", [0]*36)
        before_l, before_d = max_stacks(bl)
        after_l, after_d = max_stacks(cl)
        if root_player == "Light":
            my_stack_before, my_stack_after = before_l, after_l
            opp_stack_before, opp_stack_after = before_d, after_d
        else:
            my_stack_before, my_stack_after = before_d, after_d
            opp_stack_before, opp_stack_after = before_l, after_l

        # Mobility snapshot after the move (opponent to move)
        my_moves = GameRules.getAllLegalMoves(child)
        my_caps = sum(1 for m in my_moves if m["Direction"] in ("N","E","S","W"))
        opp_moves = GameRules.getAllLegalMoves(child)  # child already has opp to move
        opp_caps = sum(1 for m in opp_moves if m["Direction"] in ("N","E","S","W"))

        results.append({
            "move": mv,
            "score": val,
            "mat_gain": mat_gain,
            "my_stack_before": my_stack_before,
            "my_stack_after": my_stack_after,
            "opp_stack_before": opp_stack_before,
            "opp_stack_after": opp_stack_after,
            "my_moves": len(my_moves),
            "my_caps": my_caps,
            "opp_moves": len(opp_moves),
            "opp_caps": opp_caps,
        })

    # Choose best
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    best_move = best["move"]

    # Log concise summary
    _sep()
    _writeln(f"[{_ts()}] Turn={root_player}  Depth={MAX_DEPTH}")
    if SHOW_CANDIDATES:
        _writeln("Top candidates:")
        for r in results[:TOPK]:
            mv = _fmt_move(r["move"])
            _writeln(
                f"  {mv:>12}  score={r['score']:.3f}  mat+={r['mat_gain']}  "
                f"stack:{r['my_stack_before']}→{r['my_stack_after']}  "
                f"oppStack:{r['opp_stack_before']}→{r['opp_stack_after']}  "
                f"replyMob: opp {r['opp_moves']} moves / {r['opp_caps']} caps"
            )
    _writeln(f"CHOSEN MOVE: {_fmt_move(best_move)}  score={best['score']:.3f}")
    rationale = build_rationale(best)
    _writeln(f"Why: {rationale}")
    _sep()

    return best_move

############ Search ############

def alphabeta(state, depth, alpha, beta, maximizing, root_player):
    if GameRules.isGameOver(state):
        final_state = GameRules.endGame(state)
        return terminal_eval(final_state, root_player)
    if depth == 0:
        return heuristic_eval(state, root_player)

    legal = GameRules.getAllLegalMoves(state)
    if not legal:
        final_state = GameRules.endGame(state)
        return terminal_eval(final_state, root_player)

    ordered = order_moves(state, legal)
    if maximizing:
        value = -math.inf
        for mv in ordered:
            child = GameRules.playMove(state, mv)
            if child is None:
                continue
            value = max(value, alphabeta(child, depth-1, alpha, beta, False, root_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for mv in ordered:
            child = GameRules.playMove(state, mv)
            if child is None:
                continue
            value = min(value, alphabeta(child, depth-1, alpha, beta, True, root_player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

############ Heuristics ############

def heuristic_eval(state, root_player):
    light_cap = state.get("LightCapture", 0)
    dark_cap = state.get("DarkCapture", 0)
    board = state.get("Board", [0]*36)

    if root_player == "Light":
        material = light_cap - dark_cap
    else:
        material = dark_cap - light_cap

    light_max, dark_max = max_stacks(board)
    maxstack_term = (light_max - dark_max) if root_player == "Light" else (dark_max - light_max)

    my_moves = GameRules.getAllLegalMoves(state)
    my_cap_count = sum(1 for m in my_moves if m["Direction"] in ("N","E","S","W"))
    opp_state = {
        "Turn": "Dark" if state["Turn"] == "Light" else "Light",
        "LightCapture": light_cap, "DarkCapture": dark_cap,
        "Board": list(board),
    }
    opp_moves = GameRules.getAllLegalMoves(opp_state)
    opp_cap_count = sum(1 for m in opp_moves if m["Direction"] in ("N","E","S","W"))
    mobility = (0.25*len(my_moves) + 0.75*my_cap_count) - (0.25*len(opp_moves) + 0.75*opp_cap_count)

    score = (2.0*material) + (0.6*maxstack_term) + (0.5*mobility)
    if score == 0 and root_player == "Light":
        score += 0.001
    return score

def terminal_eval(state, root_player):
    l = state.get("LightCapture",0)
    d = state.get("DarkCapture",0)
    diff = (l-d) if root_player=="Light" else (d-l)
    if diff == 0:
        return 0.001 if root_player == "Light" else -0.001
    return diff

def max_stacks(board_list):
    light_max = 0
    dark_max = 0
    for idx, cnt in enumerate(board_list):
        r = idx // 6
        c = idx % 6
        if ((r+c) % 2) == 0:  # dark at even parity
            dark_max = max(dark_max, cnt)
        else:
            light_max = max(light_max, cnt)
    return light_max, dark_max

############ Ordering + Rationale ############

def order_moves(state, moves):
    before_l, before_d = max_stacks(state["Board"])
    scored = []
    for mv in moves:
        is_cap = 1 if mv["Direction"] in ("N","E","S","W") else 0
        child = GameRules.playMove(state, mv)
        if child is None:
            gain = -999
            stack_gain = -999
        else:
            if state["Turn"] == "Light":
                gain = child.get("LightCapture",0) - state.get("LightCapture",0)
            else:
                gain = child.get("DarkCapture",0) - state.get("DarkCapture",0)
            after_l, after_d = max_stacks(child["Board"])
            if state["Turn"] == "Light":
                stack_gain = (after_l - before_l) - (after_d - before_d)
            else:
                stack_gain = (after_d - before_d) - (after_l - before_l)
        scored.append((is_cap, gain, stack_gain, mv))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [t[3] for t in scored]

def build_rationale(best):
    mv = best["move"]
    mat = best["mat_gain"]
    my_stack_b, my_stack_a = best["my_stack_before"], best["my_stack_after"]
    opp_stack_b, opp_stack_a = best["opp_stack_before"], best["opp_stack_after"]
    opp_moves, opp_caps = best["opp_moves"], best["opp_caps"]

    phrases = []
    if mat > 0:
        phrases.append(f"captures {mat} piece{'s' if mat!=1 else ''}")
    if my_stack_a > my_stack_b:
        phrases.append(f"builds a larger stack ({my_stack_b}→{my_stack_a})")
    if opp_stack_a < opp_stack_b:
        phrases.append(f"shrinks opponent’s largest stack ({opp_stack_b}→{opp_stack_a})")
    if opp_caps == 0:
        phrases.append("leaves no immediate capture reply")
    elif opp_caps <= 1:
        phrases.append("limits opponent capture replies")

    core = "; ".join(phrases) if phrases else "improves position while keeping opponent’s replies manageable"
    return f"I chose {_fmt_move(mv)} because it yields the highest search score and {core}."
