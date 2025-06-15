    # futoshiki_solver.py
"""
Futoshiki Puzzle Solver and Validator
Implements the strategies from knowledge bases to solve puzzles
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from futoshiki_hard_strategies_kb import FutoshikiHardStrategiesKB

class FutoshikiSolver:
    def __init__(self):
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        
        # Grid state
        self.grid = None
        self.size = 0
        self.candidates = {}
        self.h_constraints = {}  # Horizontal constraints: (row, col) -> '<' or '>'
        self.v_constraints = {}  # Vertical constraints: (row, col) -> '<' or '>'
        
        # Solving statistics
        self.solving_stats = {
            'strategies_used': [],
            'iterations': 0,
            'cells_solved': 0
        }
    
    def solve_puzzle(self, grid: np.ndarray, h_constraints: Dict, v_constraints: Dict, 
                    allowed_strategies: List[str] = None, max_time_seconds: int = 30) -> Tuple[np.ndarray, List[str]]:
        """
        Solve Futoshiki puzzle using specified strategies
        Returns: (solved_puzzle, strategies_used)
        """
        import time
        start_time = time.time()
        
        self.grid = grid.copy()
        self.size = len(grid)
        self.h_constraints = h_constraints.copy()
        self.v_constraints = v_constraints.copy()
        
        # Reset statistics
        self.solving_stats = {'strategies_used': [], 'iterations': 0, 'cells_solved': 0}
        
        if allowed_strategies is None:
            allowed_strategies = (self.easy_kb.list_strategies() + 
                                self.moderate_kb.list_strategies() + 
                                self.hard_kb.list_strategies())
        
        # Initialize candidates
        self.initialize_candidates()
        
        max_iterations = 200
        iteration = 0
        
        while not self.is_complete() and iteration < max_iterations:
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                raise TimeoutError(f"Solver timed out after {max_time_seconds} seconds")
            
            iteration += 1
            self.solving_stats['iterations'] = iteration
            progress_made = False
            
            # Try each allowed strategy in order of complexity
            for strategy_name in self._get_strategy_order(allowed_strategies):
                if self.apply_strategy(strategy_name):
                    if strategy_name not in self.solving_stats['strategies_used']:
                        self.solving_stats['strategies_used'].append(strategy_name)
                    progress_made = True
                    break
            
            if not progress_made:
                # No more progress possible with allowed strategies
                break
        
        return self.grid, self.solving_stats['strategies_used']
    
    def _get_strategy_order(self, allowed_strategies: List[str]) -> List[str]:
        """Order strategies from simple to complex for solving efficiency"""
        easy_strategies = [s for s in allowed_strategies if s in self.easy_kb.list_strategies()]
        moderate_strategies = [s for s in allowed_strategies if s in self.moderate_kb.list_strategies()]
        hard_strategies = [s for s in allowed_strategies if s in self.hard_kb.list_strategies()]
        
        # Prioritize basic strategies first
        order = []
        
        # Core easy strategies first
        priority_easy = ['naked_single', 'constraint_propagation', 'row_uniqueness', 'column_uniqueness']
        for strategy in priority_easy:
            if strategy in easy_strategies:
                order.append(strategy)
                easy_strategies.remove(strategy)
        
        # Remaining easy strategies
        order.extend(easy_strategies)
        order.extend(moderate_strategies)
        order.extend(hard_strategies)
        
        return order
    
    def initialize_candidates(self):
        """Initialize candidate sets for each empty cell"""
        self.candidates = {}
        
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:
                    # Start with all possible values
                    candidates = set(range(1, self.size + 1))
                    
                    # Remove values already in row
                    for c in range(self.size):
                        if self.grid[row, c] != 0:
                            candidates.discard(self.grid[row, c])
                    
                    # Remove values already in column
                    for r in range(self.size):
                        if self.grid[r, col] != 0:
                            candidates.discard(self.grid[r, col])
                    
                    # Apply constraint filtering
                    candidates = self._filter_by_constraints(row, col, candidates)
                    
                    self.candidates[(row, col)] = candidates
    
    def _filter_by_constraints(self, row: int, col: int, candidates: Set[int]) -> Set[int]:
        """Filter candidates based on inequality constraints"""
        filtered = candidates.copy()
        
        # Check horizontal constraints
        # Left constraint
        if (row, col-1) in self.h_constraints:
            constraint = self.h_constraints[(row, col-1)]
            left_val = self.grid[row, col-1] if col > 0 else None
            if left_val is not None and left_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if left_val < v}
                elif constraint == '>':
                    filtered = {v for v in filtered if left_val > v}
        
        # Right constraint
        if (row, col) in self.h_constraints:
            constraint = self.h_constraints[(row, col)]
            right_val = self.grid[row, col+1] if col < self.size-1 else None
            if right_val is not None and right_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if v < right_val}
                elif constraint == '>':
                    filtered = {v for v in filtered if v > right_val}
        
        # Check vertical constraints
        # Top constraint
        if (row-1, col) in self.v_constraints:
            constraint = self.v_constraints[(row-1, col)]
            top_val = self.grid[row-1, col] if row > 0 else None
            if top_val is not None and top_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if top_val < v}
                elif constraint == '>':
                    filtered = {v for v in filtered if top_val > v}
        
        # Bottom constraint
        if (row, col) in self.v_constraints:
            constraint = self.v_constraints[(row, col)]
            bottom_val = self.grid[row+1, col] if row < self.size-1 else None
            if bottom_val is not None and bottom_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if v < bottom_val}
                elif constraint == '>':
                    filtered = {v for v in filtered if v > bottom_val}
        
        return filtered
    
    def is_complete(self) -> bool:
        """Check if puzzle is completely solved"""
        return np.all(self.grid != 0)
    
    def apply_strategy(self, strategy_name: str) -> bool:
        """Apply a specific strategy"""
        if strategy_name in self.easy_kb.list_strategies():
            return self._apply_easy_strategy(strategy_name)
        elif strategy_name in self.moderate_kb.list_strategies():
            return self._apply_moderate_strategy(strategy_name)
        elif strategy_name in self.hard_kb.list_strategies():
            return self._apply_hard_strategy(strategy_name)
        
        return False
    
    def _apply_easy_strategy(self, strategy_name: str) -> bool:
        """Apply easy strategies"""
        if strategy_name == "naked_single":
            return self._apply_naked_single()
        elif strategy_name == "constraint_propagation":
            return self._apply_constraint_propagation()
        elif strategy_name == "row_uniqueness":
            return self._apply_row_uniqueness()
        elif strategy_name == "column_uniqueness":
            return self._apply_column_uniqueness()
        elif strategy_name == "forced_by_inequality":
            return self._apply_forced_by_inequality()
        elif strategy_name == "minimum_maximum_bounds":
            return self._apply_minimum_maximum_bounds()
        elif strategy_name == "hidden_single_row":
            return self._apply_hidden_single_row()
        elif strategy_name == "hidden_single_column":
            return self._apply_hidden_single_column()
        elif strategy_name == "direct_constraint_forcing":
            return self._apply_direct_constraint_forcing()
        
        return False
    
    def _apply_moderate_strategy(self, strategy_name: str) -> bool:
        """Apply moderate strategies"""
        if strategy_name == "naked_pair":
            return self._apply_naked_pair()
        elif strategy_name == "hidden_pair":
            return self._apply_hidden_pair()
        elif strategy_name == "constraint_chain_analysis":
            return self._apply_constraint_chain_analysis()
        elif strategy_name == "constraint_splitting":
            return self._apply_constraint_splitting()
        elif strategy_name == "mutual_constraint_elimination":
            return self._apply_mutual_constraint_elimination()
        elif strategy_name == "inequality_sandwich":
            return self._apply_inequality_sandwich()
        elif strategy_name == "constraint_propagation_advanced":
            return self._apply_constraint_propagation_advanced()
        elif strategy_name == "value_forcing_by_uniqueness":
            return self._apply_value_forcing_by_uniqueness()
        
        return False
    
    def _apply_hard_strategy(self, strategy_name: str) -> bool:
        """Apply hard strategies"""
        if strategy_name == "multiple_constraint_chains":
            return self._apply_multiple_constraint_chains()
        elif strategy_name == "constraint_network_analysis":
            return self._apply_constraint_network_analysis()
        elif strategy_name == "naked_triple":
            return self._apply_naked_triple()
        elif strategy_name == "hidden_triple":
            return self._apply_hidden_triple()
        elif strategy_name == "constraint_intersection_forcing":
            return self._apply_constraint_intersection_forcing()
        elif strategy_name == "advanced_sandwich_analysis":
            return self._apply_advanced_sandwich_analysis()
        elif strategy_name == "global_constraint_consistency":
            return self._apply_global_constraint_consistency()
        elif strategy_name == "temporal_constraint_reasoning":
            return self._apply_temporal_constraint_reasoning()
        elif strategy_name == "constraint_symmetry_breaking":
            return self._apply_constraint_symmetry_breaking()
        
        return False
    
    # Easy strategy implementations
    def _apply_naked_single(self) -> bool:
        """Apply naked single strategy"""
        progress = False
        for (row, col), candidates in list(self.candidates.items()):
            if len(candidates) == 1:
                value = list(candidates)[0]
                self.grid[row, col] = value
                self._update_candidates_after_assignment(row, col, value)
                progress = True
        return progress
    
    def _apply_constraint_propagation(self) -> bool:
        """Apply basic constraint propagation"""
        progress = False
        for (row, col) in list(self.candidates.keys()):
            original_candidates = self.candidates[(row, col)].copy()
            filtered_candidates = self._filter_by_constraints(row, col, original_candidates)
            
            if len(filtered_candidates) < len(original_candidates):
                self.candidates[(row, col)] = filtered_candidates
                progress = True
                
                # Check if this leads to a naked single
                if len(filtered_candidates) == 1:
                    value = list(filtered_candidates)[0]
                    self.grid[row, col] = value
                    self._update_candidates_after_assignment(row, col, value)
        
        return progress
    
    def _apply_row_uniqueness(self) -> bool:
        """Apply row uniqueness constraint"""
        progress = False
        for row in range(self.size):
            # Get assigned values in this row
            assigned_values = set()
            for col in range(self.size):
                if self.grid[row, col] != 0:
                    assigned_values.add(self.grid[row, col])
            
            # Remove assigned values from candidates in this row
            for col in range(self.size):
                if (row, col) in self.candidates:
                    original_size = len(self.candidates[(row, col)])
                    self.candidates[(row, col)] -= assigned_values
                    if len(self.candidates[(row, col)]) < original_size:
                        progress = True
        
        return progress
    
    def _apply_column_uniqueness(self) -> bool:
        """Apply column uniqueness constraint"""
        progress = False
        for col in range(self.size):
            # Get assigned values in this column
            assigned_values = set()
            for row in range(self.size):
                if self.grid[row, col] != 0:
                    assigned_values.add(self.grid[row, col])
            
            # Remove assigned values from candidates in this column
            for row in range(self.size):
                if (row, col) in self.candidates:
                    original_size = len(self.candidates[(row, col)])
                    self.candidates[(row, col)] -= assigned_values
                    if len(self.candidates[(row, col)]) < original_size:
                        progress = True
        
        return progress
    
    def _apply_hidden_single_row(self) -> bool:
        """Apply hidden single in row strategy"""
        progress = False
        for row in range(self.size):
            # For each value, check if it can only go in one cell in this row
            for value in range(1, self.size + 1):
                possible_cells = []
                for col in range(self.size):
                    if (row, col) in self.candidates and value in self.candidates[(row, col)]:
                        possible_cells.append((row, col))
                
                if len(possible_cells) == 1:
                    r, c = possible_cells[0]
                    self.grid[r, c] = value
                    self._update_candidates_after_assignment(r, c, value)
                    progress = True
        
        return progress
    
    def _apply_hidden_single_column(self) -> bool:
        """Apply hidden single in column strategy"""
        progress = False
        for col in range(self.size):
            # For each value, check if it can only go in one cell in this column
            for value in range(1, self.size + 1):
                possible_cells = []
                for row in range(self.size):
                    if (row, col) in self.candidates and value in self.candidates[(row, col)]:
                        possible_cells.append((row, col))
                
                if len(possible_cells) == 1:
                    r, c = possible_cells[0]
                    self.grid[r, c] = value
                    self._update_candidates_after_assignment(r, c, value)
                    progress = True
        
        return progress
    
    # Placeholder implementations for moderate and hard strategies
    def _apply_forced_by_inequality(self) -> bool:
        """Apply forced by inequality strategy"""
        # Simplified implementation
        return self._apply_constraint_propagation()
    
    def _apply_minimum_maximum_bounds(self) -> bool:
        """Apply minimum/maximum bounds strategy"""
        # Simplified implementation using constraint chains
        return False
    
    def _apply_direct_constraint_forcing(self) -> bool:
        """Apply direct constraint forcing strategy"""
        # Simplified implementation
        return False
    
    def _apply_naked_pair(self) -> bool:
        """Apply naked pair strategy"""
        # Simplified implementation
        return False
    
    def _apply_hidden_pair(self) -> bool:
        """Apply hidden pair strategy"""
        # Simplified implementation
        return False
    
    def _apply_constraint_chain_analysis(self) -> bool:
        """Apply constraint chain analysis strategy"""
        # Simplified implementation
        return False
    
    def _apply_constraint_splitting(self) -> bool:
        """Apply constraint splitting strategy"""
        # Simplified implementation
        return False
    
    def _apply_mutual_constraint_elimination(self) -> bool:
        """Apply mutual constraint elimination strategy"""
        # Simplified implementation
        return False
    
    def _apply_inequality_sandwich(self) -> bool:
        """Apply inequality sandwich strategy"""
        # Simplified implementation
        return False
    
    def _apply_constraint_propagation_advanced(self) -> bool:
        """Apply advanced constraint propagation strategy"""
        # Simplified implementation
        return self._apply_constraint_propagation()
    
    def _apply_value_forcing_by_uniqueness(self) -> bool:
        """Apply value forcing by uniqueness strategy"""
        # Simplified implementation
        return False
    
    # Hard strategy placeholder implementations
    def _apply_multiple_constraint_chains(self) -> bool:
        return False
    
    def _apply_constraint_network_analysis(self) -> bool:
        return False
    
    def _apply_naked_triple(self) -> bool:
        return False
    
    def _apply_hidden_triple(self) -> bool:
        return False
    
    def _apply_constraint_intersection_forcing(self) -> bool:
        return False
    
    def _apply_advanced_sandwich_analysis(self) -> bool:
        return False
    
    def _apply_global_constraint_consistency(self) -> bool:
        return False
    
    def _apply_temporal_constraint_reasoning(self) -> bool:
        return False
    
    def _apply_constraint_symmetry_breaking(self) -> bool:
        return False
    
    def _update_candidates_after_assignment(self, row: int, col: int, value: int):
        """Update candidate sets after assigning a value to a cell"""
        # Remove from candidates
        if (row, col) in self.candidates:
            del self.candidates[(row, col)]
        
        self.solving_stats['cells_solved'] += 1
        
        # Update candidates in same row and column
        for c in range(self.size):
            if (row, c) in self.candidates:
                self.candidates[(row, c)].discard(value)
        
        for r in range(self.size):
            if (r, col) in self.candidates:
                self.candidates[(r, col)].discard(value)
        
        # Update candidates based on new constraints
        self._propagate_constraints_from_assignment(row, col, value)
    
    def _propagate_constraints_from_assignment(self, row: int, col: int, value: int):
        """Propagate constraint implications from a new assignment"""
        # Check horizontal constraints
        if (row, col) in self.h_constraints:  # Constraint to the right
            constraint = self.h_constraints[(row, col)]
            if col + 1 < self.size and (row, col + 1) in self.candidates:
                if constraint == '<':
                    # value < right_cell, so right_cell > value
                    self.candidates[(row, col + 1)] = {v for v in self.candidates[(row, col + 1)] if v > value}
                elif constraint == '>':
                    # value > right_cell, so right_cell < value
                    self.candidates[(row, col + 1)] = {v for v in self.candidates[(row, col + 1)] if v < value}
        
        if (row, col - 1) in self.h_constraints:  # Constraint from the left
            constraint = self.h_constraints[(row, col - 1)]
            if col - 1 >= 0 and (row, col - 1) in self.candidates:
                if constraint == '<':
                    # left_cell < value, so left_cell < value
                    self.candidates[(row, col - 1)] = {v for v in self.candidates[(row, col - 1)] if v < value}
                elif constraint == '>':
                    # left_cell > value, so left_cell > value
                    self.candidates[(row, col - 1)] = {v for v in self.candidates[(row, col - 1)] if v > value}
        
        # Check vertical constraints
        if (row, col) in self.v_constraints:  # Constraint below
            constraint = self.v_constraints[(row, col)]
            if row + 1 < self.size and (row + 1, col) in self.candidates:
                if constraint == '<':
                    # value < bottom_cell, so bottom_cell > value
                    self.candidates[(row + 1, col)] = {v for v in self.candidates[(row + 1, col)] if v > value}
                elif constraint == '>':
                    # value > bottom_cell, so bottom_cell < value
                    self.candidates[(row + 1, col)] = {v for v in self.candidates[(row + 1, col)] if v < value}
        
        if (row - 1, col) in self.v_constraints:  # Constraint from above
            constraint = self.v_constraints[(row - 1, col)]
            if row - 1 >= 0 and (row - 1, col) in self.candidates:
                if constraint == '<':
                    # top_cell < value, so top_cell < value
                    self.candidates[(row - 1, col)] = {v for v in self.candidates[(row - 1, col)] if v < value}
                elif constraint == '>':
                    # top_cell > value, so top_cell > value
                    self.candidates[(row - 1, col)] = {v for v in self.candidates[(row - 1, col)] if v > value}
    
    def validate_solution(self, grid: np.ndarray, h_constraints: Dict, v_constraints: Dict) -> bool:
        """Validate if the solution is correct"""
        size = len(grid)
        
        # Check if all cells are filled
        if np.any(grid == 0):
            return False
        
        # Check row uniqueness
        for row in range(size):
            if len(set(grid[row, :])) != size:
                return False
        
        # Check column uniqueness
        for col in range(size):
            if len(set(grid[:, col])) != size:
                return False
        
        # Check horizontal constraints
        for (row, col), constraint in h_constraints.items():
            if col + 1 < size:
                left_val = grid[row, col]
                right_val = grid[row, col + 1]
                if constraint == '<' and left_val >= right_val:
                    return False
                elif constraint == '>' and left_val <= right_val:
                    return False
        
        # Check vertical constraints
        for (row, col), constraint in v_constraints.items():
            if row + 1 < size:
                top_val = grid[row, col]
                bottom_val = grid[row + 1, col]
                if constraint == '<' and top_val >= bottom_val:
                    return False
                elif constraint == '>' and top_val <= bottom_val:
                    return False
        
        return True
    
    def get_solving_statistics(self) -> Dict:
        """Get solving statistics"""
        return self.solving_stats.copy()
    
    def get_difficulty_estimate(self, strategies_used: List[str]) -> str:
        """Estimate puzzle difficulty based on strategies used"""
        easy_strategies = set(self.easy_kb.list_strategies())
        moderate_strategies = set(self.moderate_kb.list_strategies())
        hard_strategies = set(self.hard_kb.list_strategies())
        
        used_strategies = set(strategies_used)
        
        if used_strategies & hard_strategies:
            return "hard"
        elif used_strategies & moderate_strategies:
            return "moderate"
        elif used_strategies & easy_strategies:
            return "easy"
        else:
            return "trivial"

    def _apply_bidirectional_constraint_chaining(self) -> bool:
            """Apply bidirectional constraint chaining strategy"""
            progress = False
            
            # Find all constraint chains
            chains = self._identify_constraint_chains()
            
            for chain in chains:
                for cell_pos in chain['cells']:
                    if cell_pos in self.candidates:
                        # Analyze constraints from both directions
                        forward_constraints = self._analyze_chain_forward(chain, cell_pos)
                        backward_constraints = self._analyze_chain_backward(chain, cell_pos)
                        
                        # Find intersection
                        intersection = forward_constraints & backward_constraints
                        
                        original_size = len(self.candidates[cell_pos])
                        self.candidates[cell_pos] &= intersection
                        
                        if len(self.candidates[cell_pos]) < original_size:
                            progress = True
            
            return progress
    
    def _apply_constraint_conflict_resolution(self) -> bool:
        """Apply constraint conflict resolution strategy"""
        progress = False
        
        # Detect conflicts in current state
        conflicts = self._detect_constraint_conflicts()
        
        if conflicts:
            # Analyze conflict sources
            conflict_graph = self._build_conflict_graph(conflicts)
            
            # Apply resolution strategies
            resolutions = self._resolve_conflicts(conflict_graph)
            
            # Apply resolutions
            for resolution in resolutions:
                if resolution['type'] == 'eliminate_candidate':
                    cell_pos = resolution['cell']
                    value = resolution['value']
                    if cell_pos in self.candidates and value in self.candidates[cell_pos]:
                        self.candidates[cell_pos].discard(value)
                        progress = True
                
                elif resolution['type'] == 'force_assignment':
                    cell_pos = resolution['cell']
                    value = resolution['value']
                    if cell_pos in self.candidates:
                        self.grid[cell_pos[0], cell_pos[1]] = value
                        self._update_candidates_after_assignment(cell_pos[0], cell_pos[1], value)
                        progress = True
        
        return progress
    
    def _identify_constraint_chains(self) -> List[Dict]:
        """Identify all constraint chains in the puzzle"""
        chains = []
        visited = set()
        
        # Find horizontal chains
        for row in range(self.size):
            chain_cells = []
            for col in range(self.size):
                if (row, col) not in visited:
                    # Start a new chain if there's a constraint
                    if (row, col) in self.h_constraints or (row, col-1) in self.h_constraints:
                        chain = self._trace_horizontal_chain(row, col, visited)
                        if len(chain) > 1:
                            chains.append({
                                'type': 'horizontal',
                                'cells': chain,
                                'constraints': self._get_chain_constraints_h(chain)
                            })
        
        # Find vertical chains
        visited.clear()
        for col in range(self.size):
            for row in range(self.size):
                if (row, col) not in visited:
                    # Start a new chain if there's a constraint
                    if (row, col) in self.v_constraints or (row-1, col) in self.v_constraints:
                        chain = self._trace_vertical_chain(row, col, visited)
                        if len(chain) > 1:
                            chains.append({
                                'type': 'vertical',
                                'cells': chain,
                                'constraints': self._get_chain_constraints_v(chain)
                            })
        
        return chains
    
    def _trace_horizontal_chain(self, start_row: int, start_col: int, visited: Set) -> List[Tuple[int, int]]:
        """Trace a horizontal constraint chain"""
        chain = []
        row = start_row
        
        # Go left to find the start
        col = start_col
        while col > 0 and (row, col-1) in self.h_constraints:
            col -= 1
        
        # Now trace right
        while col < self.size:
            chain.append((row, col))
            visited.add((row, col))
            
            if col < self.size - 1 and (row, col) in self.h_constraints:
                col += 1
            else:
                break
        
        return chain
    
    def _trace_vertical_chain(self, start_row: int, start_col: int, visited: Set) -> List[Tuple[int, int]]:
        """Trace a vertical constraint chain"""
        chain = []
        col = start_col
        
        # Go up to find the start
        row = start_row
        while row > 0 and (row-1, col) in self.v_constraints:
            row -= 1
        
        # Now trace down
        while row < self.size:
            chain.append((row, col))
            visited.add((row, col))
            
            if row < self.size - 1 and (row, col) in self.v_constraints:
                row += 1
            else:
                break
        
        return chain
    
    def _get_chain_constraints_h(self, chain: List[Tuple[int, int]]) -> List[str]:
        """Get horizontal constraints for a chain"""
        constraints = []
        for i in range(len(chain) - 1):
            row, col = chain[i]
            if (row, col) in self.h_constraints:
                constraints.append(self.h_constraints[(row, col)])
        return constraints
    
    def _get_chain_constraints_v(self, chain: List[Tuple[int, int]]) -> List[str]:
        """Get vertical constraints for a chain"""
        constraints = []
        for i in range(len(chain) - 1):
            row, col = chain[i]
            if (row, col) in self.v_constraints:
                constraints.append(self.v_constraints[(row, col)])
        return constraints
    
    def _analyze_chain_forward(self, chain: Dict, target_cell: Tuple[int, int]) -> Set[int]:
        """Analyze constraint chain in forward direction"""
        valid_values = set(range(1, self.size + 1))
        
        # Find position of target cell in chain
        try:
            target_index = chain['cells'].index(target_cell)
        except ValueError:
            return valid_values
        
        # Analyze constraints before this cell
        for i in range(target_index):
            cell_pos = chain['cells'][i]
            if self.grid[cell_pos[0], cell_pos[1]] != 0:
                # This cell is assigned, use it to constrain our target
                assigned_value = self.grid[cell_pos[0], cell_pos[1]]
                
                # Apply chain constraints between this cell and target
                for j in range(i, target_index):
                    if j < len(chain['constraints']):
                        constraint = chain['constraints'][j]
                        if constraint == '<':
                            valid_values = {v for v in valid_values if assigned_value < v}
                        elif constraint == '>':
                            valid_values = {v for v in valid_values if assigned_value > v}
                        assigned_value = None  # Reset for next iteration
        
        return valid_values
    
    def _analyze_chain_backward(self, chain: Dict, target_cell: Tuple[int, int]) -> Set[int]:
        """Analyze constraint chain in backward direction"""
        valid_values = set(range(1, self.size + 1))
        
        # Find position of target cell in chain
        try:
            target_index = chain['cells'].index(target_cell)
        except ValueError:
            return valid_values
        
        # Analyze constraints after this cell
        for i in range(target_index + 1, len(chain['cells'])):
            cell_pos = chain['cells'][i]
            if self.grid[cell_pos[0], cell_pos[1]] != 0:
                # This cell is assigned, use it to constrain our target
                assigned_value = self.grid[cell_pos[0], cell_pos[1]]
                
                # Apply chain constraints between target and this cell
                for j in range(target_index, i):
                    if j < len(chain['constraints']):
                        constraint = chain['constraints'][j]
                        if constraint == '<':
                            valid_values = {v for v in valid_values if v < assigned_value}
                        elif constraint == '>':
                            valid_values = {v for v in valid_values if v > assigned_value}
                        assigned_value = None  # Reset for next iteration
        
        return valid_values
    
    def _detect_constraint_conflicts(self) -> List[Dict]:
        """Detect conflicts in current constraint state"""
        conflicts = []
        
        # Check for cells with no valid candidates
        for cell_pos, candidates in self.candidates.items():
            if len(candidates) == 0:
                conflicts.append({
                    'type': 'empty_candidates',
                    'cell': cell_pos,
                    'description': f'Cell {cell_pos} has no valid candidates'
                })
        
        # Check for constraint violations with current assignments
        for (row, col), constraint in self.h_constraints.items():
            if col + 1 < self.size:
                left_val = self.grid[row, col]
                right_val = self.grid[row, col + 1]
                
                if left_val != 0 and right_val != 0:
                    if constraint == '<' and left_val >= right_val:
                        conflicts.append({
                            'type': 'constraint_violation',
                            'cells': [(row, col), (row, col + 1)],
                            'constraint': constraint,
                            'values': [left_val, right_val]
                        })
                    elif constraint == '>' and left_val <= right_val:
                        conflicts.append({
                            'type': 'constraint_violation',
                            'cells': [(row, col), (row, col + 1)],
                            'constraint': constraint,
                            'values': [left_val, right_val]
                        })
        
        # Check vertical constraints similarly
        for (row, col), constraint in self.v_constraints.items():
            if row + 1 < self.size:
                top_val = self.grid[row, col]
                bottom_val = self.grid[row + 1, col]
                
                if top_val != 0 and bottom_val != 0:
                    if constraint == '<' and top_val >= bottom_val:
                        conflicts.append({
                            'type': 'constraint_violation',
                            'cells': [(row, col), (row + 1, col)],
                            'constraint': constraint,
                            'values': [top_val, bottom_val]
                        })
                    elif constraint == '>' and top_val <= bottom_val:
                        conflicts.append({
                            'type': 'constraint_violation',
                            'cells': [(row, col), (row + 1, col)],
                            'constraint': constraint,
                            'values': [top_val, bottom_val]
                        })
        
        return conflicts
    
    def _build_conflict_graph(self, conflicts: List[Dict]) -> Dict:
        """Build a graph representing conflicts and their relationships"""
        conflict_graph = {
            'nodes': [],
            'edges': [],
            'conflict_clusters': []
        }
        
        # Add conflict nodes
        for i, conflict in enumerate(conflicts):
            conflict_graph['nodes'].append({
                'id': i,
                'conflict': conflict,
                'type': conflict['type']
            })
        
        # Find relationships between conflicts (cells in common)
        for i in range(len(conflicts)):
            for j in range(i + 1, len(conflicts)):
                conflict1 = conflicts[i]
                conflict2 = conflicts[j]
                
                # Check if conflicts share cells
                cells1 = set(conflict1.get('cells', [conflict1.get('cell')]))
                cells2 = set(conflict2.get('cells', [conflict2.get('cell')]))
                
                if cells1 & cells2:  # Intersection
                    conflict_graph['edges'].append({
                        'from': i,
                        'to': j,
                        'shared_cells': list(cells1 & cells2)
                    })
        
        return conflict_graph
    
    def _resolve_conflicts(self, conflict_graph: Dict) -> List[Dict]:
        """Generate resolution actions for conflicts"""
        resolutions = []
        
        for node in conflict_graph['nodes']:
            conflict = node['conflict']
            
            if conflict['type'] == 'empty_candidates':
                # Try to find why this cell has no candidates
                cell_pos = conflict['cell']
                row, col = cell_pos
                
                # Check if we can relax some constraints
                relaxable_constraints = self._find_relaxable_constraints(cell_pos)
                
                for constraint_info in relaxable_constraints:
                    resolutions.append({
                        'type': 'relax_constraint',
                        'constraint': constraint_info,
                        'cell': cell_pos
                    })
            
            elif conflict['type'] == 'constraint_violation':
                # Try to fix constraint violations
                cells = conflict['cells']
                constraint = conflict['constraint']
                values = conflict['values']
                
                # Suggest value changes
                for i, cell_pos in enumerate(cells):
                    if cell_pos in self.candidates:
                        # Find alternative values for this cell
                        current_val = values[i]
                        alternative_vals = self.candidates[cell_pos] - {current_val}
                        
                        for alt_val in alternative_vals:
                            resolutions.append({
                                'type': 'suggest_value_change',
                                'cell': cell_pos,
                                'from_value': current_val,
                                'to_value': alt_val
                            })
        
        return resolutions
    
    def _find_relaxable_constraints(self, cell_pos: Tuple[int, int]) -> List[Dict]:
        """Find constraints that could be relaxed to help resolve conflicts"""
        relaxable = []
        row, col = cell_pos
        
        # Check adjacent constraints
        adjacent_constraints = []
        
        # Left constraint
        if col > 0 and (row, col-1) in self.h_constraints:
            adjacent_constraints.append({
                'type': 'horizontal',
                'position': (row, col-1),
                'constraint': self.h_constraints[(row, col-1)],
                'affects': [(row, col-1), (row, col)]
            })
        
        # Right constraint
        if col < self.size-1 and (row, col) in self.h_constraints:
            adjacent_constraints.append({
                'type': 'horizontal',
                'position': (row, col),
                'constraint': self.h_constraints[(row, col)],
                'affects': [(row, col), (row, col+1)]
            })
        
        # Top constraint
        if row > 0 and (row-1, col) in self.v_constraints:
            adjacent_constraints.append({
                'type': 'vertical',
                'position': (row-1, col),
                'constraint': self.v_constraints[(row-1, col)],
                'affects': [(row-1, col), (row, col)]
            })
        
        # Bottom constraint
        if row < self.size-1 and (row, col) in self.v_constraints:
            adjacent_constraints.append({
                'type': 'vertical',
                'position': (row, col),
                'constraint': self.v_constraints[(row, col)],
                'affects': [(row, col), (row+1, col)]
            })
        
        return adjacent_constraints
    
    def _has_contradiction(self, test_grid: np.ndarray, test_candidates: Dict) -> bool:
        """Enhanced contradiction detection"""
        # Check for empty candidate sets
        for candidates in test_candidates.values():
            if len(candidates) == 0:
                return True
        
        # Check for constraint violations in assigned cells
        for (row, col), constraint in self.h_constraints.items():
            if col + 1 < self.size:
                left_val = test_grid[row, col]
                right_val = test_grid[row, col + 1]
                
                if left_val != 0 and right_val != 0:
                    if constraint == '<' and left_val >= right_val:
                        return True
                    elif constraint == '>' and left_val <= right_val:
                        return True
        
        for (row, col), constraint in self.v_constraints.items():
            if row + 1 < self.size:
                top_val = test_grid[row, col]
                bottom_val = test_grid[row + 1, col]
                
                if top_val != 0 and bottom_val != 0:
                    if constraint == '<' and top_val >= bottom_val:
                        return True
                    elif constraint == '>' and top_val <= bottom_val:
                        return True
        
        # Check for uniqueness violations
        for row in range(self.size):
            row_values = [test_grid[row, col] for col in range(self.size) if test_grid[row, col] != 0]
            if len(row_values) != len(set(row_values)):
                return True
        
        for col in range(self.size):
            col_values = [test_grid[row, col] for row in range(self.size) if test_grid[row, col] != 0]
            if len(col_values) != len(set(col_values)):
                return True
        
        return False    
    
    def _apply_hard_strategy(self, strategy_name: str) -> bool:
        """Apply hard strategies with full implementations"""
        strategy_methods = {
            "multiple_constraint_chains": self._apply_multiple_constraint_chains,
            "constraint_network_analysis": self._apply_constraint_network_analysis,
            "naked_triple": self._apply_naked_triple,
            "hidden_triple": self._apply_hidden_triple,
            "constraint_intersection_forcing": self._apply_constraint_intersection_forcing,
            "advanced_sandwich_analysis": self._apply_advanced_sandwich_analysis,
            "global_constraint_consistency": self._apply_global_constraint_consistency,
            "temporal_constraint_reasoning": self._apply_temporal_constraint_reasoning,
            "constraint_symmetry_breaking": self._apply_constraint_symmetry_breaking,
            "constraint_forcing_networks": self._apply_constraint_forcing_networks,
            "advanced_constraint_propagation": self._apply_advanced_constraint_propagation,
            "bidirectional_constraint_chaining": self._apply_bidirectional_constraint_chaining,
            "constraint_conflict_resolution": self._apply_constraint_conflict_resolution
        }
        
        method = strategy_methods.get(strategy_name)
        if method:
            return method()
        return False
    
    # Hard strategy implementations
    def _apply_multiple_constraint_chains(self) -> bool:
        """Apply multiple constraint chains strategy"""
        progress = False
        
        # Find cells that are part of multiple constraint chains
        multi_chain_cells = self._find_multi_chain_cells()
        
        for cell_pos in multi_chain_cells:
            if cell_pos in self.candidates:
                original_candidates = self.candidates[cell_pos].copy()
                
                # Analyze constraints from multiple chains
                chain_constraints = self._get_chain_constraints(cell_pos)
                
                # Find intersection of valid values from all chains
                valid_values = set(range(1, self.size + 1))
                for chain_constraint in chain_constraints:
                    chain_valid = self._get_valid_values_for_chain(cell_pos, chain_constraint)
                    valid_values &= chain_valid
                
                # Update candidates
                self.candidates[cell_pos] &= valid_values
                
                if len(self.candidates[cell_pos]) < len(original_candidates):
                    progress = True
        
        return progress
    
    def _apply_constraint_network_analysis(self) -> bool:
        """Apply constraint network analysis strategy"""
        progress = False
        
        # Build constraint network
        network = self._build_constraint_network()
        
        # Analyze network for forced assignments
        forced_assignments = self._analyze_network_for_forced_values(network)
        
        for (row, col), value in forced_assignments:
            if (row, col) in self.candidates:
                self.grid[row, col] = value
                self._update_candidates_after_assignment(row, col, value)
                progress = True
        
        return progress
    
    def _apply_naked_triple(self) -> bool:
        """Apply naked triple strategy"""
        progress = False
        
        # Check rows
        for row in range(self.size):
            progress |= self._find_naked_triples_in_unit(
                [(row, col) for col in range(self.size) if (row, col) in self.candidates]
            )
        
        # Check columns
        for col in range(self.size):
            progress |= self._find_naked_triples_in_unit(
                [(row, col) for row in range(self.size) if (row, col) in self.candidates]
            )
        
        return progress
    
    def _apply_hidden_triple(self) -> bool:
        """Apply hidden triple strategy"""
        progress = False
        
        # Check rows
        for row in range(self.size):
            progress |= self._find_hidden_triples_in_unit(
                [(row, col) for col in range(self.size) if (row, col) in self.candidates]
            )
        
        # Check columns
        for col in range(self.size):
            progress |= self._find_hidden_triples_in_unit(
                [(row, col) for row in range(self.size) if (row, col) in self.candidates]
            )
        
        return progress
    
    def _apply_constraint_intersection_forcing(self) -> bool:
        """Apply constraint intersection forcing strategy"""
        progress = False
        
        # Find constraint intersections
        intersections = self._find_constraint_intersections()
        
        for intersection_point, chains in intersections.items():
            if intersection_point in self.candidates:
                # Calculate valid values from intersection of all chains
                intersection_values = set(range(1, self.size + 1))
                
                for chain in chains:
                    chain_values = self._get_valid_values_for_chain(intersection_point, chain)
                    intersection_values &= chain_values
                
                original_size = len(self.candidates[intersection_point])
                self.candidates[intersection_point] &= intersection_values
                
                if len(self.candidates[intersection_point]) < original_size:
                    progress = True
        
        return progress
    
    def _apply_advanced_sandwich_analysis(self) -> bool:
        """Apply advanced sandwich analysis strategy"""
        progress = False
        
        # Find cells with multiple adjacent constraints
        multi_constrained_cells = self._find_multi_constrained_cells()
        
        for cell_pos in multi_constrained_cells:
            if cell_pos in self.candidates:
                row, col = cell_pos
                original_candidates = self.candidates[cell_pos].copy()
                
                # Analyze all adjacent constraints
                valid_values = set()
                
                for value in original_candidates:
                    if self._satisfies_all_adjacent_constraints(row, col, value):
                        valid_values.add(value)
                
                self.candidates[cell_pos] = valid_values
                
                if len(valid_values) < len(original_candidates):
                    progress = True
        
        return progress
    
    def _apply_global_constraint_consistency(self) -> bool:
        """Apply global constraint consistency strategy"""
        progress = False
        
        # Use constraint satisfaction techniques
        # This is a simplified implementation
        for (row, col) in list(self.candidates.keys()):
            original_candidates = self.candidates[(row, col)].copy()
            
            # Check each candidate for global consistency
            consistent_values = set()
            
            for value in original_candidates:
                if self._is_globally_consistent(row, col, value):
                    consistent_values.add(value)
            
            self.candidates[(row, col)] = consistent_values
            
            if len(consistent_values) < len(original_candidates):
                progress = True
        
        return progress
    
    def _apply_temporal_constraint_reasoning(self) -> bool:
        """Apply temporal constraint reasoning strategy"""
        # This is a complex strategy that would require forward search
        # Simplified implementation
        return False
    
    def _apply_constraint_symmetry_breaking(self) -> bool:
        """Apply constraint symmetry breaking strategy"""
        # This would require symmetry detection and breaking
        # Simplified implementation
        return False
    
    def _apply_constraint_forcing_networks(self) -> bool:
        """Apply constraint forcing networks strategy"""
        progress = False
        
        # Build forcing network for each cell
        for (row, col) in list(self.candidates.keys()):
            if len(self.candidates[(row, col)]) > 1:
                # Analyze forcing implications
                forced_value = self._analyze_forcing_network(row, col)
                
                if forced_value is not None:
                    self.grid[row, col] = forced_value
                    self._update_candidates_after_assignment(row, col, forced_value)
                    progress = True
        
        return progress
    
    def _apply_advanced_constraint_propagation(self) -> bool:
        """Apply advanced constraint propagation strategy"""
        progress = False
        max_iterations = 10
        
        for iteration in range(max_iterations):
            iteration_progress = False
            
            # Apply multiple levels of propagation
            iteration_progress |= self._apply_constraint_propagation()
            iteration_progress |= self._apply_mutual_constraint_elimination()
            iteration_progress |= self._apply_constraint_chain_analysis()
            
            if iteration_progress:
                progress = True
            else:
                break  # No more progress
        
        return progress
    
    # Helper methods for hard strategies
    def _find_multi_chain_cells(self) -> List[Tuple[int, int]]:
        """Find cells that are part of multiple constraint chains"""
        multi_chain_cells = []
        
        for row in range(self.size):
            for col in range(self.size):
                if (row, col) in self.candidates:
                    chain_count = 0
                    
                    # Check horizontal chains
                    if self._is_part_of_horizontal_chain(row, col):
                        chain_count += 1
                    
                    # Check vertical chains
                    if self._is_part_of_vertical_chain(row, col):
                        chain_count += 1
                    
                    if chain_count > 1:
                        multi_chain_cells.append((row, col))
        
        return multi_chain_cells
    
    def _is_part_of_horizontal_chain(self, row: int, col: int) -> bool:
        """Check if cell is part of a horizontal constraint chain"""
        # Check if there are constraints to the left or right
        has_left_constraint = (row, col-1) in self.h_constraints if col > 0 else False
        has_right_constraint = (row, col) in self.h_constraints if col < self.size-1 else False
        
        return has_left_constraint or has_right_constraint
    
    def _is_part_of_vertical_chain(self, row: int, col: int) -> bool:
        """Check if cell is part of a vertical constraint chain"""
        # Check if there are constraints above or below
        has_top_constraint = (row-1, col) in self.v_constraints if row > 0 else False
        has_bottom_constraint = (row, col) in self.v_constraints if row < self.size-1 else False
        
        return has_top_constraint or has_bottom_constraint
    
    def _get_chain_constraints(self, cell_pos: Tuple[int, int]) -> List[Dict]:
        """Get all constraint chains affecting a cell"""
        row, col = cell_pos
        chains = []
        
        # Horizontal chain
        if self._is_part_of_horizontal_chain(row, col):
            chains.append({'type': 'horizontal', 'row': row, 'col': col})
        
        # Vertical chain
        if self._is_part_of_vertical_chain(row, col):
            chains.append({'type': 'vertical', 'row': row, 'col': col})
        
        return chains
    
    def _get_valid_values_for_chain(self, cell_pos: Tuple[int, int], chain: Dict) -> Set[int]:
        """Get valid values for a cell based on a constraint chain"""
        row, col = cell_pos
        valid_values = set(range(1, self.size + 1))
        
        if chain['type'] == 'horizontal':
            # Check horizontal constraints
            valid_values = self._filter_by_constraints(row, col, valid_values)
        elif chain['type'] == 'vertical':
            # Check vertical constraints
            valid_values = self._filter_by_constraints(row, col, valid_values)
        
        return valid_values
    
    def _build_constraint_network(self) -> Dict:
        """Build a network representation of all constraints"""
        network = {
            'nodes': [],
            'edges': [],
            'constraints': []
        }
        
        # Add nodes for each cell
        for row in range(self.size):
            for col in range(self.size):
                network['nodes'].append((row, col))
        
        # Add edges for constraints
        for (row, col), constraint in self.h_constraints.items():
            if col + 1 < self.size:
                network['edges'].append({
                    'from': (row, col),
                    'to': (row, col + 1),
                    'constraint': constraint,
                    'type': 'horizontal'
                })
        
        for (row, col), constraint in self.v_constraints.items():
            if row + 1 < self.size:
                network['edges'].append({
                    'from': (row, col),
                    'to': (row + 1, col),
                    'constraint': constraint,
                    'type': 'vertical'
                })
        
        return network
    
    def _analyze_network_for_forced_values(self, network: Dict) -> List[Tuple[Tuple[int, int], int]]:
        """Analyze constraint network to find forced value assignments"""
        forced_assignments = []
        
        # This is a simplified analysis
        for (row, col) in self.candidates:
            if len(self.candidates[(row, col)]) == 1:
                value = list(self.candidates[(row, col)])[0]
                forced_assignments.append(((row, col), value))
        
        return forced_assignments
    
    def _find_naked_triples_in_unit(self, unit_cells: List[Tuple[int, int]]) -> bool:
        """Find naked triples within a unit"""
        progress = False
        
        # Find cells with 2 or 3 candidates
        candidate_cells = [cell for cell in unit_cells 
                          if 2 <= len(self.candidates.get(cell, set())) <= 3]
        
        # Look for triples
        for i in range(len(candidate_cells)):
            for j in range(i + 1, len(candidate_cells)):
                for k in range(j + 1, len(candidate_cells)):
                    cell1, cell2, cell3 = candidate_cells[i], candidate_cells[j], candidate_cells[k]
                    
                    combined_candidates = (self.candidates[cell1] | 
                                         self.candidates[cell2] | 
                                         self.candidates[cell3])
                    
                    if len(combined_candidates) == 3:
                        # Found a naked triple - eliminate from other cells
                        for cell in unit_cells:
                            if cell not in [cell1, cell2, cell3] and cell in self.candidates:
                                before_size = len(self.candidates[cell])
                                self.candidates[cell] -= combined_candidates
                                if len(self.candidates[cell]) < before_size:
                                    progress = True
        
        return progress
    
    def _find_hidden_triples_in_unit(self, unit_cells: List[Tuple[int, int]]) -> bool:
        """Find hidden triples within a unit"""
        progress = False
        
        # For each combination of three values, check if they can only go in three cells
        for val1 in range(1, self.size + 1):
            for val2 in range(val1 + 1, self.size + 1):
                for val3 in range(val2 + 1, self.size + 1):
                    
                    cells_with_values = []
                    for cell in unit_cells:
                        if cell in self.candidates:
                            cell_candidates = self.candidates[cell]
                            if {val1, val2, val3} & cell_candidates:
                                cells_with_values.append(cell)
                    
                    if len(cells_with_values) == 3:
                        # Check if these values can only go in these three cells
                        valid_triple = True
                        for other_cell in unit_cells:
                            if other_cell not in cells_with_values and other_cell in self.candidates:
                                if {val1, val2, val3} & self.candidates[other_cell]:
                                    valid_triple = False
                                    break
                        
                        if valid_triple:
                            # Found hidden triple - remove other candidates
                            for cell in cells_with_values:
                                before_size = len(self.candidates[cell])
                                self.candidates[cell] &= {val1, val2, val3}
                                if len(self.candidates[cell]) < before_size:
                                    progress = True
        
        return progress
    
    def _find_constraint_intersections(self) -> Dict[Tuple[int, int], List]:
        """Find points where multiple constraint chains intersect"""
        intersections = {}
        
        for row in range(self.size):
            for col in range(self.size):
                chains = []
                
                if self._is_part_of_horizontal_chain(row, col):
                    chains.append({'type': 'horizontal', 'row': row, 'col': col})
                
                if self._is_part_of_vertical_chain(row, col):
                    chains.append({'type': 'vertical', 'row': row, 'col': col})
                
                if len(chains) > 1:
                    intersections[(row, col)] = chains
        
        return intersections
    
    def _find_multi_constrained_cells(self) -> List[Tuple[int, int]]:
        """Find cells with multiple adjacent constraints"""
        multi_constrained = []
        
        for row in range(self.size):
            for col in range(self.size):
                constraint_count = 0
                
                # Check all adjacent positions for constraints
                # Left constraint
                if col > 0 and (row, col-1) in self.h_constraints:
                    constraint_count += 1
                
                # Right constraint
                if col < self.size-1 and (row, col) in self.h_constraints:
                    constraint_count += 1
                
                # Top constraint
                if row > 0 and (row-1, col) in self.v_constraints:
                    constraint_count += 1
                
                # Bottom constraint
                if row < self.size-1 and (row, col) in self.v_constraints:
                    constraint_count += 1
                
                if constraint_count >= 2:
                    multi_constrained.append((row, col))
        
        return multi_constrained
    
    def _satisfies_all_adjacent_constraints(self, row: int, col: int, value: int) -> bool:
        """Check if a value satisfies all adjacent constraints for a cell"""
        # Check left constraint
        if col > 0 and (row, col-1) in self.h_constraints:
            constraint = self.h_constraints[(row, col-1)]
            left_val = self.grid[row, col-1]
            if left_val != 0:
                if constraint == '<' and left_val >= value:
                    return False
                elif constraint == '>' and left_val <= value:
                    return False
        
        # Check right constraint
        if col < self.size-1 and (row, col) in self.h_constraints:
            constraint = self.h_constraints[(row, col)]
            right_val = self.grid[row, col+1]
            if right_val != 0:
                if constraint == '<' and value >= right_val:
                    return False
                elif constraint == '>' and value <= right_val:
                    return False
        
        # Check top constraint
        if row > 0 and (row-1, col) in self.v_constraints:
            constraint = self.v_constraints[(row-1, col)]
            top_val = self.grid[row-1, col]
            if top_val != 0:
                if constraint == '<' and top_val >= value:
                    return False
                elif constraint == '>' and top_val <= value:
                    return False
        
        # Check bottom constraint
        if row < self.size-1 and (row, col) in self.v_constraints:
            constraint = self.v_constraints[(row, col)]
            bottom_val = self.grid[row+1, col]
            if bottom_val != 0:
                if constraint == '<' and value >= bottom_val:
                    return False
                elif constraint == '>' and value <= bottom_val:
                    return False
        
        return True
    
    def _is_globally_consistent(self, row: int, col: int, value: int) -> bool:
        """Check if assigning a value is globally consistent with all constraints"""
        # Make a temporary assignment
        original_val = self.grid[row, col]
        self.grid[row, col] = value
        
        # Check if this creates any conflicts
        consistent = True
        
        # Check row uniqueness
        row_values = [self.grid[row, c] for c in range(self.size) if self.grid[row, c] != 0]
        if len(row_values) != len(set(row_values)):
            consistent = False
        
        # Check column uniqueness
        if consistent:
            col_values = [self.grid[r, col] for r in range(self.size) if self.grid[r, col] != 0]
            if len(col_values) != len(set(col_values)):
                consistent = False
        
        # Check constraint satisfaction
        if consistent:
            consistent = self._check_all_constraints_satisfied()
        
        # Restore original value
        self.grid[row, col] = original_val
        
        return consistent
    
    def _check_all_constraints_satisfied(self) -> bool:
        """Check if all constraints are currently satisfied"""
        # Check horizontal constraints
        for (row, col), constraint in self.h_constraints.items():
            if col + 1 < self.size:
                left_val = self.grid[row, col]
                right_val = self.grid[row, col + 1]
                
                if left_val != 0 and right_val != 0:
                    if constraint == '<' and left_val >= right_val:
                        return False
                    elif constraint == '>' and left_val <= right_val:
                        return False
        
        # Check vertical constraints
        for (row, col), constraint in self.v_constraints.items():
            if row + 1 < self.size:
                top_val = self.grid[row, col]
                bottom_val = self.grid[row + 1, col]
                
                if top_val != 0 and bottom_val != 0:
                    if constraint == '<' and top_val >= bottom_val:
                        return False
                    elif constraint == '>' and top_val <= bottom_val:
                        return False
        
        return True
    
    def _analyze_forcing_network(self, row: int, col: int) -> Optional[int]:
        """Analyze forcing network to determine if a value is forced"""
        candidates = self.candidates[(row, col)].copy()
        
        if len(candidates) <= 1:
            return None
        
        forced_value = None
        
        # For each candidate, see if assuming it leads to a unique solution
        for value in candidates:
            # Create a copy of the current state
            test_grid = self.grid.copy()
            test_candidates = {k: v.copy() for k, v in self.candidates.items()}
            
            # Assign the test value
            test_grid[row, col] = value
            del test_candidates[(row, col)]
            
            # Propagate constraints from this assignment
            self._propagate_test_assignment(test_grid, test_candidates, row, col, value)
            
            # Check if this leads to a contradiction
            if self._has_contradiction(test_grid, test_candidates):
                # This value leads to contradiction, so it can't be correct
                continue
            else:
                # This value is potentially valid
                if forced_value is None:
                    forced_value = value
                else:
                    # Multiple values are valid, so nothing is forced
                    return None
        
        return forced_value
    
    def _propagate_test_assignment(self, test_grid: np.ndarray, test_candidates: Dict, 
                                  row: int, col: int, value: int):
        """Propagate constraints from a test assignment"""
        # Remove value from row and column candidates
        for c in range(self.size):
            if (row, c) in test_candidates:
                test_candidates[(row, c)].discard(value)
        
        for r in range(self.size):
            if (r, col) in test_candidates:
                test_candidates[(r, col)].discard(value)
        
        # Apply constraint propagation
        self._propagate_constraints_in_test(test_grid, test_candidates, row, col, value)
    
    def _propagate_constraints_in_test(self, test_grid: np.ndarray, test_candidates: Dict,
                                      row: int, col: int, value: int):
        """Propagate constraints in test scenario"""
        # Check horizontal constraints
        if (row, col) in self.h_constraints:  # Constraint to the right
            constraint = self.h_constraints[(row, col)]
            if col + 1 < self.size and (row, col + 1) in test_candidates:
                if constraint == '<':
                    test_candidates[(row, col + 1)] = {v for v in test_candidates[(row, col + 1)] if v > value}
                elif constraint == '>':
                    test_candidates[(row, col + 1)] = {v for v in test_candidates[(row, col + 1)] if v < value}
        
        if (row, col - 1) in self.h_constraints:  # Constraint from the left
            constraint = self.h_constraints[(row, col - 1)]
            if col - 1 >= 0 and (row, col - 1) in test_candidates:
                if constraint == '<':
                    test_candidates[(row, col - 1)] = {v for v in test_candidates[(row, col - 1)] if v < value}
                elif constraint == '>':
                    test_candidates[(row, col - 1)] = {v for v in test_candidates[(row, col - 1)] if v > value}
        
        # Check vertical constraints (similar logic)
        if (row, col) in self.v_constraints:  # Constraint below
            constraint = self.v_constraints[(row, col)]
            if row + 1 < self.size and (row + 1, col) in test_candidates:
                if constraint == '<':
                    test_candidates[(row + 1, col)] = {v for v in test_candidates[(row + 1, col)] if v > value}
                elif constraint == '>':
                    test_candidates[(row + 1, col)] = {v for v in test_candidates[(row + 1, col)] if v < value}
        
        if (row - 1, col) in self.v_constraints:  # Constraint from above
            constraint = self.v_constraints[(row - 1, col)]
            if row - 1 >= 0 and (row - 1, col) in test_candidates:
                if constraint == '<':
                    test_candidates[(row - 1, col)] = {v for v in test_candidates[(row - 1, col)] if v < value}
                elif constraint == '>':
                    test_candidates[(row - 1, col)] = {v for v in test_candidates[(row - 1, col)] if v > value}
    
    def _has_contradiction(self, test_grid: np.ndarray, test_candidates: Dict) -> bool:
        """Check if the test state has any contradictions"""
        # Check for empty candidate sets
        for candidates in test_candidates.values():
            if len(candidates) == 0:
                return True
        
        # Check for constraint violations in assigned cells
        for (row, col), constraint in self.h_constraints.items():
            if col + 1 < self.size:
                left_val = test_grid[row, col]
                right_val = test_grid[row, col + 1]
                
                if left_val != 0 and right_val != 0:
                    if constraint == '<' and left_val >= right_val:
                        return True
                    elif constraint == '>' and left_val <= right_val:
                        return True
        
        for (row, col), constraint in self.v_constraints.items():
            if row + 1 < self.size:
                top_val = test_grid[row, col]
                bottom_val = test_grid[row + 1, col]
                
                if top_val != 0 and bottom_val != 0:
                    if constraint == '<' and top_val >= bottom_val:
                        return True
                    elif constraint == '>' and top_val <= bottom_val:
                        return True
        
        return False
