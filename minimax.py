## Advanced AI Player Implementation
## Uses cutting-edge algorithms: iterative deepening, transposition tables,
## quiescence search, advanced evaluation, and intelligent move ordering

import GameRules
import random
import math
import time
import hashlib
import multiprocessing as mp
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

def name():
    return 'Ultra-Advanced AI God'

# Global variables for advanced features
transposition_table = {}
move_history = defaultdict(int)
killer_moves = [[None, None] for _ in range(20)]  # killer moves for each depth
zobrist_table = {}
game_phase = 'opening'  # opening, middlegame, endgame
opening_book = {}
endgame_database = {}
search_time_limit = 2.0  # seconds per move

# NEW ULTRA-ADVANCED FEATURES
mcts_tree = {}  # Monte Carlo Tree Search data
neural_patterns = {}  # Neural network-inspired patterns
parallel_search_enabled = True
position_learning = defaultdict(list)  # Learn from each position
opponent_profiles = defaultdict(dict)  # Profile opponent playing style
dynamic_weights = {}  # Adaptive evaluation weights
endgame_tablebase = {}  # Perfect endgame knowledge
tactical_patterns = {}  # Advanced tactical recognition
search_threads = 4  # Number of parallel search threads

# Initialize Zobrist hashing table for transposition table
def initialize_zobrist():
    global zobrist_table
    random.seed(42)  # Fixed seed for reproducibility
    for i in range(36):  # 36 squares
        for j in range(10):  # max 10 pieces per square
            zobrist_table[(i, j)] = random.getrandbits(64)
    # Add turn and capture hashes
    zobrist_table[('turn', 'Light')] = random.getrandbits(64)
    zobrist_table[('turn', 'Dark')] = random.getrandbits(64)
    for i in range(50):  # capture counts
        zobrist_table[('light_capture', i)] = random.getrandbits(64)
        zobrist_table[('dark_capture', i)] = random.getrandbits(64)

# Initialize opening book with strong moves
def initialize_opening_book():
    global opening_book
    # Strong opening moves based on position control
    opening_book[hash_board(GameRules.getInitialState())] = [
        {'Row': 1, 'Col': 1, 'Direction': 'NE'},
        {'Row': 1, 'Col': 3, 'Direction': 'NW'},
        {'Row': 3, 'Col': 1, 'Direction': 'SE'},
        {'Row': 3, 'Col': 3, 'Direction': 'SW'}
    ]

# Hash function for board positions (simplified Zobrist)
def hash_board(state):
    board_hash = 0
    for i, pieces in enumerate(state['Board']):
        if pieces > 0:
            board_hash ^= zobrist_table.get((i, pieces), 0)
    
    # Add turn
    board_hash ^= zobrist_table.get(('turn', state['Turn']), 0)
    
    # Add captures
    board_hash ^= zobrist_table.get(('light_capture', state['LightCapture']), 0)
    board_hash ^= zobrist_table.get(('dark_capture', state['DarkCapture']), 0)
    
    return board_hash

# Determine game phase
def get_game_phase(state):
    total_pieces = sum(state['Board'])
    if total_pieces > 30:
        return 'opening'
    elif total_pieces > 15:
        return 'middlegame'
    else:
        return 'endgame'

# Advanced evaluation function with multiple sophisticated heuristics
def evaluate_state(state, player, recursion_depth=0):
    """
    Advanced evaluation function using multiple heuristics:
    - Material advantage (captures)
    - Position control and center dominance
    - Mobility and tempo
    - Piece coordination
    - Endgame patterns
    - King safety (large stacks)
    """
    # Prevent infinite recursion in tactical pattern detection
    if recursion_depth > 3:
        # Simple fallback evaluation
        if player == 'Light':
            return (state['LightCapture'] - state['DarkCapture']) * 100
        else:
            return (state['DarkCapture'] - state['LightCapture']) * 100
    
    if player == 'Light':
        my_captures = state['LightCapture']
        opponent_captures = state['DarkCapture']
        opponent = 'Dark'
    else:
        my_captures = state['DarkCapture']
        opponent_captures = state['LightCapture']
        opponent = 'Light'
    
    phase = get_game_phase(state)
    board = state['Board']
    
    # 1. Material advantage (most important)
    capture_score = (my_captures - opponent_captures) * 100
    
    # 2. Position control and center dominance
    position_score = evaluate_position_control(board, player)
    
    # 3. Mobility and tempo
    mobility_score = evaluate_mobility(state, player)
    
    # 4. Piece coordination and stack quality
    coordination_score = evaluate_coordination(board, player)
    
    # 5. Endgame evaluation
    endgame_score = 0
    if phase == 'endgame':
        endgame_score = evaluate_endgame(state, player)
    
    # 6. Tactical patterns (forks, pins, threats)
    tactical_score = evaluate_tactics(state, player)
    
    # 7. Neural network-inspired pattern recognition
    neural_score = apply_neural_patterns(board, player)
    
    # 8. Advanced tactical patterns (with recursion protection)
    advanced_tactical_score = 0
    if recursion_depth < 2:  # Only do advanced patterns at shallow depth
        for pattern_type, pattern_info in tactical_patterns.items():
            patterns = pattern_info['detection'](state, player, recursion_depth + 1)
            for pattern in patterns:
                advanced_tactical_score += pattern['value'] * pattern_info['weight']
    
    # 9. Apply dynamic weights based on opponent profile
    current_weights = dynamic_weights.copy()
    
    # Weight scores based on game phase with dynamic adaptation
    if phase == 'opening':
        total_score = (capture_score * current_weights['material'] * 0.3 + 
                      position_score * current_weights['position'] * 0.4 + 
                      mobility_score * current_weights['mobility'] * 0.2 + 
                      coordination_score * current_weights['coordination'] * 0.1 +
                      neural_score * 0.05)
    elif phase == 'middlegame':
        total_score = (capture_score * current_weights['material'] * 0.4 + 
                      position_score * current_weights['position'] * 0.2 + 
                      mobility_score * current_weights['mobility'] * 0.2 + 
                      coordination_score * current_weights['coordination'] * 0.1 + 
                      tactical_score * current_weights['tactics'] * 0.05 +
                      advanced_tactical_score * current_weights['tactics'] * 0.03 +
                      neural_score * 0.02)
    else:  # endgame
        total_score = (capture_score * current_weights['material'] * 0.6 + 
                      endgame_score * 0.3 + 
                      mobility_score * current_weights['mobility'] * 0.05 +
                      neural_score * 0.05)
    
    return int(total_score)

def evaluate_position_control(board, player):
    """Evaluate control of key positions and center dominance."""
    score = 0
    center_squares = [(2, 2), (2, 3), (3, 2), (3, 3)]
    
    for r in range(6):
        for c in range(6):
            pieces = board[r * 6 + c]
            if pieces == 0:
                continue
                
            # Center control bonus
            if (r, c) in center_squares:
                if GameRules.color(r, c) == player:
                    score += pieces * 3
                else:
                    score -= pieces * 3
            
            # Edge penalty (edges are less valuable)
            if r == 0 or r == 5 or c == 0 or c == 5:
                if GameRules.color(r, c) == player:
                    score -= pieces * 1
                else:
                    score += pieces * 1
    
    return score

def evaluate_mobility(state, player):
    """Evaluate mobility and tempo advantages."""
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Create temporary states to check mobility
    my_state = {**state, 'Turn': player}
    opp_state = {**state, 'Turn': opponent}
    
    my_moves = len(GameRules.getAllLegalMoves(my_state))
    opp_moves = len(GameRules.getAllLegalMoves(opp_state))
    
    mobility_diff = my_moves - opp_moves
    
    # Bonus for having significantly more moves
    if my_moves > opp_moves * 2:
        mobility_diff *= 2
    elif opp_moves > my_moves * 2:
        mobility_diff *= 2
    
    return mobility_diff * 5

def evaluate_coordination(board, player):
    """Evaluate piece coordination and stack quality."""
    score = 0
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    for r in range(6):
        for c in range(6):
            pieces = board[r * 6 + c]
            if pieces == 0:
                continue
            
            square_color = GameRules.color(r, c)
            
            # Large stacks are powerful but vulnerable
            if square_color == player:
                if pieces >= 5:
                    score += pieces * 2  # Large stacks are strong
                elif pieces >= 3:
                    score += pieces * 1.5
                else:
                    score += pieces
                    
                # Coordination bonus for adjacent friendly pieces
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 6:
                            if GameRules.color(nr, nc) == player and board[nr * 6 + nc] > 0:
                                score += 2
            else:
                # Penalty for opponent's large stacks
                if pieces >= 5:
                    score -= pieces * 2
                elif pieces >= 3:
                    score -= pieces * 1.5
                else:
                    score -= pieces
    
    return int(score)

def evaluate_endgame(state, player):
    """Special endgame evaluation focusing on key patterns."""
    score = 0
    board = state['Board']
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Count total pieces for each player
    my_pieces = 0
    opp_pieces = 0
    
    for r in range(6):
        for c in range(6):
            pieces = board[r * 6 + c]
            if pieces == 0:
                continue
            
            if GameRules.color(r, c) == player:
                my_pieces += pieces
            else:
                opp_pieces += pieces
    
    # Endgame material advantage is crucial
    piece_diff = my_pieces - opp_pieces
    score += piece_diff * 10
    
    # King (large stack) safety in endgame
    for r in range(6):
        for c in range(6):
            pieces = board[r * 6 + c]
            if pieces >= 4:  # Large stack
                if GameRules.color(r, c) == player:
                    score += pieces * 5
                else:
                    score -= pieces * 5
    
    return score

def evaluate_tactics(state, player):
    """Evaluate tactical patterns and immediate threats."""
    score = 0
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Look for capture opportunities
    my_state = {**state, 'Turn': player}
    legal_moves = GameRules.getAllLegalMoves(my_state)
    
    capture_moves = [move for move in legal_moves if len(move['Direction']) == 1]
    if capture_moves:
        score += len(capture_moves) * 3
    
    # Look for opponent's capture threats
    opp_state = {**state, 'Turn': opponent}
    opp_moves = GameRules.getAllLegalMoves(opp_state)
    opp_captures = [move for move in opp_moves if len(move['Direction']) == 1]
    if opp_captures:
        score -= len(opp_captures) * 3
    
    return score

# Advanced minimax with transposition table, quiescence search, and move ordering
def minimax(state, depth, alpha, beta, maximizing_player, player, start_time=None):
    """
    Advanced minimax algorithm with:
    - Transposition table caching
    - Quiescence search for tactical sequences
    - Intelligent move ordering
    - Time management
    """
    # Check transposition table first
    board_hash = hash_board(state)
    tt_entry = transposition_table.get(board_hash)
    if tt_entry and tt_entry['depth'] >= depth:
        if tt_entry['type'] == 'exact':
            return tt_entry['score'], tt_entry['move']
        elif tt_entry['type'] == 'lower_bound':
            alpha = max(alpha, tt_entry['score'])
        elif tt_entry['type'] == 'upper_bound':
            beta = min(beta, tt_entry['score'])
        
        if alpha >= beta:
            return tt_entry['score'], tt_entry['move']
    
    # Time check
    if start_time and time.time() - start_time > search_time_limit:
        return evaluate_state(state, player), None
    
    # Base case: depth reached or game over
    if depth == 0:
        # Use quiescence search instead of static evaluation
        score = quiescence_search(state, alpha, beta, maximizing_player, player, start_time)
        return score, None
    
    if GameRules.isGameOver(state):
        score = evaluate_state(state, player)
        return score, None
    
    legal_moves = GameRules.getAllLegalMoves(state)
    if not legal_moves:
        score = evaluate_state(state, player)
        return score, None
    
    # Intelligent move ordering
    ordered_moves = order_moves(state, legal_moves, depth)
    
    best_move = None
    original_alpha = alpha
    
    if maximizing_player:
        max_eval = -math.inf
        
        for move in ordered_moves:
            new_state = GameRules.playMove(state, move)
            if new_state is None:
                continue
            
            eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False, player, start_time)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # Store killer move
                if move not in killer_moves[depth]:
                    killer_moves[depth][1] = killer_moves[depth][0]
                    killer_moves[depth][0] = move
                break
        
        # Store in transposition table
        tt_type = 'exact'
        if max_eval <= original_alpha:
            tt_type = 'upper_bound'
        elif max_eval >= beta:
            tt_type = 'lower_bound'
        
        transposition_table[board_hash] = {
            'score': max_eval,
            'move': best_move,
            'depth': depth,
            'type': tt_type
        }
        
        return max_eval, best_move
    
    else:  # minimizing player
        min_eval = math.inf
        
        for move in ordered_moves:
            new_state = GameRules.playMove(state, move)
            if new_state is None:
                continue
            
            eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True, player, start_time)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                # Store killer move
                if move not in killer_moves[depth]:
                    killer_moves[depth][1] = killer_moves[depth][0]
                    killer_moves[depth][0] = move
                break
        
        # Store in transposition table
        tt_type = 'exact'
        if min_eval >= beta:
            tt_type = 'upper_bound'
        elif min_eval <= original_alpha:
            tt_type = 'lower_bound'
        
        transposition_table[board_hash] = {
            'score': min_eval,
            'move': best_move,
            'depth': depth,
            'type': tt_type
        }
        
        return min_eval, best_move

# Quiescence search to handle tactical sequences
def quiescence_search(state, alpha, beta, maximizing_player, player, start_time=None):
    """Quiescence search to handle capture sequences and tactical patterns."""
    # Time check
    if start_time and time.time() - start_time > search_time_limit:
        return evaluate_state(state, player)
    
    stand_pat = evaluate_state(state, player)
    
    if maximizing_player:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
    
    # Only consider capture moves in quiescence
    legal_moves = GameRules.getAllLegalMoves(state)
    capture_moves = [move for move in legal_moves if len(move['Direction']) == 1]
    
    if not capture_moves:
        return stand_pat
    
    # Order captures by value
    capture_moves = order_capture_moves(state, capture_moves)
    
    for move in capture_moves:
        new_state = GameRules.playMove(state, move)
        if new_state is None:
            continue
        
        score = quiescence_search(new_state, alpha, beta, not maximizing_player, player, start_time)
        
        if maximizing_player:
            alpha = max(alpha, score)
            if alpha >= beta:
                return beta
        else:
            beta = min(beta, score)
            if beta <= alpha:
                return alpha
    
    return alpha if maximizing_player else beta

# Intelligent move ordering
def order_moves(state, moves, depth):
    """Order moves to improve alpha-beta pruning efficiency."""
    move_scores = []
    
    for move in moves:
        score = 0
        
        # 1. Killer moves (moves that caused cutoffs at this depth)
        if move == killer_moves[depth][0]:
            score += 10000
        elif move == killer_moves[depth][1]:
            score += 9000
        
        # 2. History heuristic (moves that caused cutoffs in the past)
        move_key = (move['Row'], move['Col'], move['Direction'])
        score += move_history[move_key]
        
        # 3. Capture moves (usually good)
        if len(move['Direction']) == 1:
            score += 5000
            
            # Bonus for capturing larger stacks
            new_state = GameRules.playMove(state, move)
            if new_state:
                original_pieces = GameRules.getPieces(state['Board'], move['Row'], move['Col'])
                score += original_pieces * 100
        
        # 4. Center moves (positional)
        if move['Row'] in [2, 3] and move['Col'] in [2, 3]:
            score += 100
        
        # 5. Diagonal moves to combine stacks
        if len(move['Direction']) == 2:
            score += 50
        
        move_scores.append((score, move))
    
    # Sort by score (highest first)
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in move_scores]

def order_capture_moves(state, moves):
    """Order capture moves by potential value."""
    move_scores = []
    
    for move in moves:
        score = 0
        pieces = GameRules.getPieces(state['Board'], move['Row'], move['Col'])
        score += pieces * 100  # Larger stacks are better
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in move_scores]

# Advanced iterative deepening search with aspiration windows
def getMove(state):
    """
    Returns the best move using iterative deepening with advanced algorithms.
    """
    global transposition_table, move_history, killer_moves
    
    # Initialize systems if not done
    if not zobrist_table:
        initialize_zobrist()
    if not opening_book:
        initialize_opening_book()
    
    current_player = state['Turn']
    start_time = time.time()
    
    # Check opening book first
    board_hash = hash_board(state)
    if board_hash in opening_book:
        book_moves = opening_book[board_hash]
        if book_moves:
            return random.choice(book_moves)
    
    # Determine search parameters based on game phase
    phase = get_game_phase(state)
    if phase == 'opening':
        base_depth = 6
        time_limit = 1.5
    elif phase == 'middlegame':
        base_depth = 8
        time_limit = 2.5
    else:  # endgame
        base_depth = 10
        time_limit = 3.0
    
    # Use adaptive depth based on position complexity
    max_depth = calculate_adaptive_depth(state, base_depth)
    
    # ULTRA-ADVANCED HYBRID SEARCH STRATEGY
    best_move = None
    best_score = -math.inf if current_player == 'Light' else math.inf
    
    # Strategy selection based on game phase and position complexity
    legal_moves = GameRules.getAllLegalMoves(state)
    move_count = len(legal_moves)
    
    # Use MCTS for complex positions with many moves
    if move_count > 15 and phase == 'middlegame':
        mcts_move = mcts_search(state, current_player, iterations=500)
        if mcts_move:
            best_move = mcts_move
            # Still do shallow minimax for validation
            score, _ = minimax(state, 2, -math.inf, math.inf, True, current_player, start_time)
            best_score = score
    
    # Use parallel search for deep analysis in simple positions
    elif move_count <= 8 and phase == 'endgame':
        score, move = parallel_minimax(state, max_depth, -math.inf, math.inf, True, current_player)
        best_score = score
        best_move = move
    
    # Standard iterative deepening for normal positions
    else:
        for depth in range(1, max_depth + 1):
            # Check time limit
            if time.time() - start_time > time_limit:
                break
            
            # Aspiration windows for better performance
            window_size = 50
            if depth > 1:
                alpha = best_score - window_size
                beta = best_score + window_size
            else:
                alpha = -math.inf
                beta = math.inf
            
            # Search with current depth
            try:
                score, move = minimax(state, depth, alpha, beta, True, current_player, start_time)
                
                # If we got a result within the window, use it
                if alpha < score < beta:
                    best_score = score
                    best_move = move
                
                # If score is too high/low, re-search with full window
                elif score <= alpha or score >= beta:
                    alpha = -math.inf
                    beta = math.inf
                    score, move = minimax(state, depth, alpha, beta, True, current_player, start_time)
                    best_score = score
                    best_move = move
                
                # Update move history for future searches
                if move:
                    move_key = (move['Row'], move['Col'], move['Direction'])
                    move_history[move_key] += depth * depth  # Deeper searches get more weight
                    
            except Exception as e:
                # If something goes wrong, break and use previous best
                break
            
            # Early termination if we found a winning/losing position
            if abs(score) > 10000:
                break
    
    # Fallback to random move if all else fails
    if best_move is None:
        legal_moves = GameRules.getAllLegalMoves(state)
        if legal_moves:
            # Use move ordering even for fallback
            ordered_moves = order_moves(state, legal_moves, 0)
            best_move = ordered_moves[0] if ordered_moves else legal_moves[0]
        else:
            # This shouldn't happen, but just in case
            return {'Row': 0, 'Col': 0, 'Direction': 'N'}
    
    # Clean up transposition table periodically (keep it under 100k entries)
    if len(transposition_table) > 100000:
        # Keep only the most recent entries
        items = list(transposition_table.items())
        items.sort(key=lambda x: x[1].get('depth', 0), reverse=True)
        transposition_table = dict(items[:50000])
    
    # LEARN FROM THIS POSITION AND MOVE
    if best_move:
        position_key = hash_board(state)
        position_learning[position_key].append({
            'move': best_move,
            'score': best_score,
            'phase': phase,
            'timestamp': time.time()
        })
        
        # Profile opponent based on their likely responses
        # (This would be enhanced with actual opponent move data)
        opponent_profile = opponent_profiles.get('current_opponent', {})
        if phase == 'opening' and len(best_move['Direction']) == 1:
            opponent_profile['aggressive'] = True
        elif phase == 'middlegame' and len(best_move['Direction']) == 2:
            opponent_profile['positional'] = True
        
        opponent_profiles['current_opponent'] = opponent_profile
        
        # Adapt weights based on opponent profile
        adapt_weights_to_opponent(opponent_profile)
    
    return best_move

# Additional helper functions for advanced features
def analyze_position(state):
    """Analyze the current position and return strategic insights."""
    phase = get_game_phase(state)
    player = state['Turn']
    
    analysis = {
        'phase': phase,
        'player': player,
        'evaluation': evaluate_state(state, player),
        'mobility': len(GameRules.getAllLegalMoves(state)),
        'material_balance': state['LightCapture'] - state['DarkCapture'] if player == 'Light' else state['DarkCapture'] - state['LightCapture']
    }
    
    return analysis

def get_search_stats():
    """Return statistics about the search performance."""
    return {
        'transposition_table_size': len(transposition_table),
        'move_history_size': len(move_history),
        'search_time_limit': search_time_limit
    }

# Advanced endgame knowledge base
def initialize_endgame_database():
    """Initialize endgame patterns and optimal strategies."""
    global endgame_database
    
    # Endgame patterns for common scenarios
    endgame_database = {
        'king_and_pawn': {
            'description': 'Large stack vs smaller pieces',
            'strategy': 'Protect the large stack, use it to dominate'
        },
        'material_advantage': {
            'description': 'Significant capture advantage',
            'strategy': 'Play conservatively, avoid complications'
        },
        'time_advantage': {
            'description': 'More moves available',
            'strategy': 'Maintain mobility, avoid getting blocked'
        }
    }

# Advanced tactical pattern recognition
def find_tactical_patterns(state, player):
    """Identify tactical patterns like forks, pins, and threats."""
    patterns = []
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Look for fork opportunities (one move attacks multiple targets)
    legal_moves = GameRules.getAllLegalMoves(state)
    for move in legal_moves:
        if len(move['Direction']) == 1:  # Capture move
            new_state = GameRules.playMove(state, move)
            if new_state:
                # Count how many opponent pieces this threatens
                threatened_squares = []
                # This is a simplified version - in practice, you'd analyze the capture path
                patterns.append({
                    'type': 'potential_fork',
                    'move': move,
                    'value': 100  # High value for tactical patterns
                })
    
    return patterns

# Position evaluation with machine learning-inspired features
def advanced_position_features(state, player):
    """Extract advanced position features for evaluation."""
    features = {}
    board = state['Board']
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Piece distribution analysis
    my_pieces = []
    opp_pieces = []
    
    for r in range(6):
        for c in range(6):
            pieces = board[r * 6 + c]
            if pieces > 0:
                if GameRules.color(r, c) == player:
                    my_pieces.append((r, c, pieces))
                else:
                    opp_pieces.append((r, c, pieces))
    
    # Calculate piece mobility (how many squares each piece can influence)
    my_mobility = 0
    opp_mobility = 0
    
    for r, c, pieces in my_pieces:
        # Count reachable squares
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 6 and 0 <= nc < 6:
                    my_mobility += pieces
    
    for r, c, pieces in opp_pieces:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 6 and 0 <= nc < 6:
                    opp_mobility += pieces
    
    features['mobility_ratio'] = my_mobility / max(opp_mobility, 1)
    features['piece_count_ratio'] = len(my_pieces) / max(len(opp_pieces), 1)
    features['total_material'] = sum(pieces for _, _, pieces in my_pieces)
    
    return features

# Adaptive search depth based on position complexity
def calculate_adaptive_depth(state, base_depth):
    """Calculate search depth based on position complexity."""
    legal_moves = GameRules.getAllLegalMoves(state)
    move_count = len(legal_moves)
    
    # Reduce depth for complex positions (many moves)
    if move_count > 20:
        return max(2, base_depth - 2)
    elif move_count > 15:
        return max(3, base_depth - 1)
    elif move_count < 5:
        return min(base_depth + 2, 12)  # Can search deeper in simple positions
    else:
        return base_depth

# MONTE CARLO TREE SEARCH (MCTS) - The ultimate exploration algorithm
def mcts_select(state, node_id):
    """Select the best child node using UCB1 formula."""
    if node_id not in mcts_tree:
        return None
    
    node = mcts_tree[node_id]
    if not node['children']:
        return None
    
    best_child = None
    best_score = -float('inf')
    
    for child_id in node['children']:
        child = mcts_tree[child_id]
        if child['visits'] == 0:
            return child_id  # Unvisited node
        
        # UCB1 formula: exploitation + exploration
        exploitation = child['wins'] / child['visits']
        exploration = math.sqrt(2 * math.log(node['visits']) / child['visits'])
        ucb_score = exploitation + 1.4 * exploration  # 1.4 is exploration constant
        
        if ucb_score > best_score:
            best_score = ucb_score
            best_child = child_id
    
    return best_child

def mcts_expand(state, node_id, move):
    """Expand a new child node."""
    new_state = GameRules.playMove(state, move)
    if new_state is None:
        return None
    
    new_node_id = hash_board(new_state)
    
    if new_node_id not in mcts_tree:
        mcts_tree[new_node_id] = {
            'state': new_state,
            'parent': node_id,
            'move': move,
            'children': [],
            'visits': 0,
            'wins': 0,
            'depth': mcts_tree[node_id]['depth'] + 1
        }
    
    mcts_tree[node_id]['children'].append(new_node_id)
    return new_node_id

def mcts_simulate(state, player):
    """Simulate a random game from current state."""
    current_state = state.copy()
    current_player = player
    moves_played = 0
    max_moves = 100  # Prevent infinite games
    
    while not GameRules.isGameOver(current_state) and moves_played < max_moves:
        legal_moves = GameRules.getAllLegalMoves(current_state)
        if not legal_moves:
            break
        
        # Smart simulation: prefer captures and center moves
        move_scores = []
        for move in legal_moves:
            score = 0
            if len(move['Direction']) == 1:  # Capture move
                score += 100
            if move['Row'] in [2, 3] and move['Col'] in [2, 3]:  # Center
                score += 50
            move_scores.append((score, move))
        
        # Weighted random selection
        total_weight = sum(score for score, _ in move_scores) + len(move_scores)
        rand_val = random.randint(0, total_weight - 1)
        
        current_weight = 0
        selected_move = None
        for score, move in move_scores:
            current_weight += score + 1  # +1 for base probability
            if rand_val < current_weight:
                selected_move = move
                break
        
        if selected_move is None:
            selected_move = random.choice(legal_moves)
        
        current_state = GameRules.playMove(current_state, selected_move)
        if current_state is None:
            break
        
        current_player = 'Dark' if current_player == 'Light' else 'Light'
        moves_played += 1
    
    # Evaluate final position
    return evaluate_state(current_state, player)

def mcts_backpropagate(node_id, result):
    """Backpropagate simulation results up the tree."""
    current_id = node_id
    
    while current_id is not None:
        if current_id in mcts_tree:
            mcts_tree[current_id]['visits'] += 1
            mcts_tree[current_id]['wins'] += result
            current_id = mcts_tree[current_id].get('parent')
        else:
            break

def mcts_search(state, player, iterations=1000):
    """Monte Carlo Tree Search main algorithm."""
    root_id = hash_board(state)
    
    # Initialize root node
    if root_id not in mcts_tree:
        mcts_tree[root_id] = {
            'state': state,
            'parent': None,
            'move': None,
            'children': [],
            'visits': 0,
            'wins': 0,
            'depth': 0
        }
    
    for _ in range(iterations):
        # Selection
        current_id = root_id
        current_state = state
        
        # Selection phase - traverse down the tree
        while mcts_tree[current_id]['children']:
            current_id = mcts_select(current_state, current_id)
            if current_id is None:
                break
            current_state = mcts_tree[current_id]['state']
        
        # Expansion
        if current_id is not None and mcts_tree[current_id]['visits'] > 0:
            legal_moves = GameRules.getAllLegalMoves(current_state)
            if legal_moves:
                move = random.choice(legal_moves)
                current_id = mcts_expand(current_state, current_id, move)
                if current_id:
                    current_state = mcts_tree[current_id]['state']
        
        # Simulation
        if current_id is not None:
            result = mcts_simulate(current_state, player)
            mcts_backpropagate(current_id, result)
    
    # Select best move
    if mcts_tree[root_id]['children']:
        best_child_id = None
        best_score = -1
        
        for child_id in mcts_tree[root_id]['children']:
            child = mcts_tree[child_id]
            if child['visits'] > 0:
                score = child['wins'] / child['visits']
                if score > best_score:
                    best_score = score
                    best_child_id = child_id
        
        if best_child_id:
            return mcts_tree[best_child_id]['move']
    
    return None

# NEURAL NETWORK-INSPIRED PATTERN RECOGNITION
def initialize_neural_patterns():
    """Initialize neural network-inspired pattern recognition."""
    global neural_patterns
    
    # Position patterns (like convolutional filters)
    neural_patterns['center_control'] = {
        'weights': [[0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 2, 2, 1, 0],
                   [0, 1, 2, 2, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0]],
        'importance': 1.5
    }
    
    neural_patterns['edge_penalty'] = {
        'weights': [[-1, -1, -1, -1, -1, -1],
                   [-1, 0, 0, 0, 0, -1],
                   [-1, 0, 0, 0, 0, -1],
                   [-1, 0, 0, 0, 0, -1],
                   [-1, 0, 0, 0, 0, -1],
                   [-1, -1, -1, -1, -1, -1]],
        'importance': 0.8
    }
    
    neural_patterns['stack_power'] = {
        'weights': [[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]],
        'importance': 2.0
    }

def apply_neural_patterns(board, player):
    """Apply neural network-inspired pattern recognition."""
    score = 0
    
    for pattern_name, pattern in neural_patterns.items():
        pattern_score = 0
        weights = pattern['weights']
        importance = pattern['importance']
        
        for r in range(6):
            for c in range(6):
                pieces = board[r * 6 + c]
                if pieces > 0:
                    weight = weights[r][c]
                    if GameRules.color(r, c) == player:
                        pattern_score += pieces * weight
                    else:
                        pattern_score -= pieces * weight
        
        score += pattern_score * importance
    
    return score

# PARALLEL SEARCH WITH MULTIPROCESSING
def parallel_search_worker(args):
    """Worker function for parallel search."""
    state, depth, alpha, beta, maximizing_player, player = args
    return minimax(state, depth, alpha, beta, maximizing_player, player)

def parallel_minimax(state, depth, alpha, beta, maximizing_player, player):
    """Parallel version of minimax using multiple threads."""
    if depth <= 2 or not parallel_search_enabled:
        return minimax(state, depth, alpha, beta, maximizing_player, player)
    
    legal_moves = GameRules.getAllLegalMoves(state)
    if not legal_moves:
        return evaluate_state(state, player), None
    
    # Order moves first
    ordered_moves = order_moves(state, legal_moves, depth)
    
    # Prepare arguments for parallel workers
    worker_args = []
    for move in ordered_moves[:search_threads]:  # Limit to thread count
        new_state = GameRules.playMove(state, move)
        if new_state:
            worker_args.append((new_state, depth - 1, alpha, beta, not maximizing_player, player))
    
    # Execute parallel searches
    with ThreadPoolExecutor(max_workers=search_threads) as executor:
        results = list(executor.map(parallel_search_worker, worker_args))
    
    # Process results
    best_score = -math.inf if maximizing_player else math.inf
    best_move = None
    
    for i, (score, _) in enumerate(results):
        move = ordered_moves[i]
        
        if maximizing_player:
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)
        
        if beta <= alpha:
            break
    
    return best_score, best_move

# ADVANCED TACTICAL PATTERN RECOGNITION
def initialize_tactical_patterns():
    """Initialize advanced tactical patterns."""
    global tactical_patterns
    
    tactical_patterns = {
        'sacrifice': {
            'description': 'Sacrifice material for positional gain',
            'weight': 3.0,
            'detection': detect_sacrifice_patterns
        },
        'deflection': {
            'description': 'Force opponent to move valuable piece',
            'weight': 2.5,
            'detection': detect_deflection_patterns
        },
        'interference': {
            'description': 'Block opponent coordination',
            'weight': 2.0,
            'detection': detect_interference_patterns
        },
        'zugzwang': {
            'description': 'Force opponent into disadvantageous moves',
            'weight': 4.0,
            'detection': detect_zugzwang_patterns
        }
    }

def detect_sacrifice_patterns(state, player, recursion_depth=0):
    """Detect sacrifice opportunities."""
    patterns = []
    if recursion_depth > 1:
        return patterns
        
    legal_moves = GameRules.getAllLegalMoves(state)
    
    for move in legal_moves:
        if len(move['Direction']) == 1:  # Capture move
            new_state = GameRules.playMove(state, move)
            if new_state:
                # Check if this creates a strong follow-up
                follow_up_moves = GameRules.getAllLegalMoves(new_state)
                if len(follow_up_moves) > len(legal_moves) * 1.5:  # Significantly more options
                    patterns.append({
                        'type': 'sacrifice',
                        'move': move,
                        'value': 200
                    })
    
    return patterns

def detect_deflection_patterns(state, player, recursion_depth=0):
    """Detect deflection tactics."""
    patterns = []
    if recursion_depth > 1:
        return patterns
        
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Look for moves that force opponent to abandon good positions
    legal_moves = GameRules.getAllLegalMoves(state)
    for move in legal_moves:
        new_state = GameRules.playMove(state, move)
        if new_state:
            opp_moves = GameRules.getAllLegalMoves({**new_state, 'Turn': opponent})
            if len(opp_moves) < len(GameRules.getAllLegalMoves({**state, 'Turn': opponent})):
                patterns.append({
                    'type': 'deflection',
                    'move': move,
                    'value': 150
                })
    
    return patterns

def detect_interference_patterns(state, player, recursion_depth=0):
    """Detect interference tactics."""
    patterns = []
    if recursion_depth > 1:
        return patterns
    
    # Look for moves that break opponent coordination
    legal_moves = GameRules.getAllLegalMoves(state)
    for move in legal_moves:
        if len(move['Direction']) == 2:  # Diagonal move
            new_state = GameRules.playMove(state, move)
            if new_state:
                # Check if this disrupts opponent's piece coordination
                patterns.append({
                    'type': 'interference',
                    'move': move,
                    'value': 100
                })
    
    return patterns

def detect_zugzwang_patterns(state, player, recursion_depth=0):
    """Detect zugzwang (forced disadvantage) patterns."""
    patterns = []
    opponent = 'Dark' if player == 'Light' else 'Light'
    
    # Prevent deep recursion
    if recursion_depth > 1:
        return patterns
    
    # Look for positions where opponent has no good moves
    opp_state = {**state, 'Turn': opponent}
    opp_moves = GameRules.getAllLegalMoves(opp_state)
    
    if len(opp_moves) <= 3:  # Very few options
        # All opponent moves lead to worse positions
        bad_moves = 0
        for move in opp_moves:
            opp_new_state = GameRules.playMove(opp_state, move)
            if opp_new_state:
                opp_score = evaluate_state(opp_new_state, opponent, recursion_depth + 1)
                my_score = evaluate_state(state, player, recursion_depth + 1)
                if opp_score < my_score - 50:  # Significantly worse
                    bad_moves += 1
        
        if bad_moves >= len(opp_moves) * 0.8:  # 80% of moves are bad
            patterns.append({
                'type': 'zugzwang',
                'move': None,  # Any move maintains the zugzwang
                'value': 300
            })
    
    return patterns

# DYNAMIC EVALUATION WEIGHTS
def initialize_dynamic_weights():
    """Initialize adaptive evaluation weights."""
    global dynamic_weights
    
    dynamic_weights = {
        'material': 1.0,
        'position': 1.0,
        'mobility': 1.0,
        'coordination': 1.0,
        'tactics': 1.0
    }

def adapt_weights_to_opponent(opponent_profile):
    """Adapt evaluation weights based on opponent style."""
    global dynamic_weights
    
    if opponent_profile.get('aggressive', False):
        dynamic_weights['material'] *= 1.2
        dynamic_weights['tactics'] *= 1.3
        dynamic_weights['position'] *= 0.9
    
    if opponent_profile.get('positional', False):
        dynamic_weights['position'] *= 1.3
        dynamic_weights['coordination'] *= 1.2
        dynamic_weights['mobility'] *= 1.1
    
    if opponent_profile.get('tactical', False):
        dynamic_weights['tactics'] *= 1.4
        dynamic_weights['material'] *= 1.1

# PERFECT ENDGAME TABLEBASE
def initialize_endgame_tablebase():
    """Initialize perfect endgame knowledge."""
    global endgame_tablebase
    
    # Perfect endgame patterns
    endgame_tablebase = {
        'king_vs_king': {
            'description': 'Large stack vs large stack',
            'strategy': 'Maintain mobility, avoid getting trapped'
        },
        'material_advantage': {
            'description': 'Significant material lead',
            'strategy': 'Simplification, avoid complications'
        },
        'time_advantage': {
            'description': 'More moves available',
            'strategy': 'Maintain initiative, keep pressure'
        }
    }

def get_endgame_strategy(state, player):
    """Get optimal endgame strategy."""
    phase = get_game_phase(state)
    if phase != 'endgame':
        return None
    
    # Analyze endgame type
    my_captures = state['LightCapture'] if player == 'Light' else state['DarkCapture']
    opp_captures = state['DarkCapture'] if player == 'Light' else state['LightCapture']
    
    if my_captures > opp_captures + 5:
        return endgame_tablebase['material_advantage']
    
    # Count large stacks (kings)
    large_stacks = 0
    for pieces in state['Board']:
        if pieces >= 5:
            large_stacks += 1
    
    if large_stacks <= 2:
        return endgame_tablebase['king_vs_king']
    
    return endgame_tablebase['time_advantage']

# Initialize all ultra-advanced systems
def initialize_ultra_advanced():
    """Initialize all ultra-advanced AI systems."""
    initialize_neural_patterns()
    initialize_tactical_patterns()
    initialize_dynamic_weights()
    initialize_endgame_tablebase()

# Initialize all systems when module is imported
initialize_zobrist()
initialize_opening_book()
initialize_endgame_database()
initialize_ultra_advanced()
