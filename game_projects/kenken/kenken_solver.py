# kenken_solver.py
"""
Ken Ken Puzzle Solver and Validator
Implements the strategies from knowledge bases to solve Ken Ken puzzles
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from itertools import permutations, combinations_with_replacement
import math

from kenken_easy_strategies_kb import KenKenEasyStrategiesKB
from kenken_moderate_strategies_kb import KenKenModerateStrategiesKB
from kenken_hard_strategies_kb import KenKenHardStrategiesKB

class KenKenSolver:
    def __init__(self):
        self.easy_kb = KenKenEasyStrategiesKB()
        self.moderate_kb = KenKenModerateStrategiesKB()
        self.hard_kb = KenKenHardStrategiesKB()
        
        # Solver state
        self.grid = None
        self.grid_size = 0
        self.cages = []
        self.candidates = {}
        self.solved_cells = set()
        
    def solve_puzzle(self, grid: np.ndarray, cages: List[Dict], allowed_strategies: List[str] = None, 
                    max_time_seconds: int = 60) -> Tuple[np.ndarray, List[str]]:
        """
        Solve Ken Ken puzzle using specified strategies with time limit
        Returns: (solved_puzzle, strategies_used)
        """
        import time
        start_time = time.time()
        
        if allowed_strategies is None:
            allowed_strategies = (self.easy_kb.list_strategies() + 
                                self.moderate_kb.list_strategies() + 
                                self.hard_kb.list_strategies())
        
        # Initialize solver state
        self.grid = grid.copy()
        self.grid_size = len(grid)
        self.cages = cages
        self.initialize_candidates()
        
        strategies_used = []
        max_iterations = 200
        iteration = 0
        
        while not self.is_complete() and iteration < max_iterations:
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                raise TimeoutError(f"Solver timed out after {max_time_seconds} seconds")
            
            iteration += 1
            progress_made = False
            
            # Try each allowed strategy
            for strategy_name in allowed_strategies:
                if self.apply_strategy(strategy_name):
                    strategies_used.append(strategy_name)
                    progress_made = True
                    break  # Try again from the beginning with updated state
            
            if not progress_made:
                # No more progress possible with allowed strategies
                break
        
        return self.grid, strategies_used
    
    def initialize_candidates(self):
        """Initialize candidate sets for each empty cell"""
        self.candidates = {}
        self.solved_cells = set()
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] == 0:
                    self.candidates[(row, col)] = self.get_possible_values(row, col)
                else:
                    self.solved_cells.add((row, col))
    
    def get_possible_values(self, row: int, col: int) -> Set[int]:
        """Get possible values for a cell based on Latin square and cage constraints"""
        if self.grid[row, col] != 0:
            return set()
        
        # Start with all values 1 to grid_size
        possible = set(range(1, self.grid_size + 1))
        
        # Remove values already in the same row
        for c in range(self.grid_size):
            if self.grid[row, c] != 0:
                possible.discard(self.grid[row, c])
        
        # Remove values already in the same column
        for r in range(self.grid_size):
            if self.grid[r, col] != 0:
                possible.discard(self.grid[r, col])
        
        # Apply cage constraints
        cell_cage = self.get_cell_cage(row, col)
        if cell_cage:
            cage_possible = self.get_cage_possible_values(cell_cage, row, col)
            possible = possible.intersection(cage_possible)
        
        return possible
    
    def get_cell_cage(self, row: int, col: int) -> Optional[Dict]:
        """Find the cage containing the given cell"""
        for cage in self.cages:
            if (row, col) in cage['cells']:
                return cage
        return None
    
    def get_cage_possible_values(self, cage: Dict, target_row: int, target_col: int) -> Set[int]:
        """Get possible values for a specific cell within a cage"""
        cage_cells = cage['cells']
        operation = cage['operation']
        target = cage['target']
        
        # Get currently assigned values in the cage
        assigned_values = []
        empty_cells = []
        
        for row, col in cage_cells:
            if self.grid[row, col] != 0:
                assigned_values.append(self.grid[row, col])
            else:
                empty_cells.append((row, col))
        
        # If this is the only empty cell, calculate its value directly
        if len(empty_cells) == 1 and (target_row, target_col) == empty_cells[0]:
            return self.calculate_final_cage_value(cage, assigned_values)
        
        # Otherwise, find all possible combinations that could work
        return self.find_possible_values_in_cage(cage, assigned_values, empty_cells, target_row, target_col)
    
    def calculate_final_cage_value(self, cage: Dict, assigned_values: List[int]) -> Set[int]:
        """Calculate the required value for the last empty cell in a cage"""
        operation = cage['operation']
        target = cage['target']
        
        if operation == 'add':
            current_sum = sum(assigned_values)
            required_value = target - current_sum
            if 1 <= required_value <= self.grid_size:
                return {required_value}
        
        elif operation == 'subtract':
            if len(assigned_values) == 1:
                # Two possibilities: assigned_value - x = target or x - assigned_value = target
                val = assigned_values[0]
                possibilities = {val + target, val - target}
                return {v for v in possibilities if 1 <= v <= self.grid_size}
        
        elif operation == 'multiply':
            current_product = 1
            for val in assigned_values:
                current_product *= val
            if target % current_product == 0:
                required_value = target // current_product
                if 1 <= required_value <= self.grid_size:
                    return {required_value}
        
        elif operation == 'divide':
            if len(assigned_values) == 1:
                val = assigned_values[0]
                possibilities = {val * target, val // target if val % target == 0 else None}
                return {v for v in possibilities if v and 1 <= v <= self.grid_size}
        
        return set()
    
    def find_possible_values_in_cage(self, cage: Dict, assigned_values: List[int], 
                                   empty_cells: List[Tuple[int, int]], target_row: int, target_col: int) -> Set[int]:
        """Find possible values for a cell when multiple cells are empty"""
        operation = cage['operation']
        target = cage['target']
        num_empty = len(empty_cells)
        
        possible_values = set()
        
        # Try all possible combinations for empty cells
        for values in combinations_with_replacement(range(1, self.grid_size + 1), num_empty):
            for perm in permutations(values):
                # Check if this assignment satisfies the cage constraint
                test_values = assigned_values + list(perm)
                if self.evaluate_cage_constraint(test_values, operation, target):
                    # Find which value would go in our target cell
                    target_index = empty_cells.index((target_row, target_col))
                    possible_values.add(perm[target_index])
        
        return possible_values
    
    def evaluate_cage_constraint(self, values: List[int], operation: str, target: int) -> bool:
        """Check if a set of values satisfies a cage constraint"""
        if not values:
            return False
        
        if operation == 'add':
            return sum(values) == target
        
        elif operation == 'subtract':
            if len(values) == 2:
                return abs(values[0] - values[1]) == target
            return False
        
        elif operation == 'multiply':
            result = 1
            for val in values:
                result *= val
            return result == target
        
        elif operation == 'divide':
            if len(values) == 2:
                return max(values) / min(values) == target
            return False
        
        return False
    
    def update_candidates_after_assignment(self, row: int, col: int, value: int):
        """Update candidate sets after assigning a value to a cell"""
        # Remove from candidates of this cell
        if (row, col) in self.candidates:
            del self.candidates[(row, col)]
        
        self.solved_cells.add((row, col))
        
        # Update candidates in same row
        for c in range(self.grid_size):
            if (row, c) in self.candidates:
                self.candidates[(row, c)].discard(value)
        
        # Update candidates in same column
        for r in range(self.grid_size):
            if (r, col) in self.candidates:
                self.candidates[(r, col)].discard(value)
        
        # Update candidates in same cage
        cell_cage = self.get_cell_cage(row, col)
        if cell_cage:
            for cage_row, cage_col in cell_cage['cells']:
                if (cage_row, cage_col) in self.candidates:
                    new_possible = self.get_cage_possible_values(cell_cage, cage_row, cage_col)
                    self.candidates[(cage_row, cage_col)] = self.candidates[(cage_row, cage_col)].intersection(new_possible)
    
    def is_complete(self) -> bool:
        """Check if puzzle is completely solved"""
        return np.all(self.grid != 0)
    
    def apply_strategy(self, strategy_name: str) -> bool:
        """Apply a specific strategy"""
        
        # Easy strategies
        if strategy_name == "single_cell_cage":
            return self.apply_single_cell_cage()
        elif strategy_name == "simple_addition_cage":
            return self.apply_simple_addition_cage()
        elif strategy_name == "simple_subtraction_cage":
            return self.apply_simple_subtraction_cage()
        elif strategy_name == "naked_single":
            return self.apply_naked_single()
        elif strategy_name == "hidden_single_row":
            return self.apply_hidden_single_row()
        elif strategy_name == "hidden_single_column":
            return self.apply_hidden_single_column()
        elif strategy_name == "cage_completion":
            return self.apply_cage_completion()
        elif strategy_name == "basic_multiplication_cage":
            return self.apply_basic_multiplication_cage()
        elif strategy_name == "basic_division_cage":
            return self.apply_basic_division_cage()
        
        # Moderate strategies
        elif strategy_name == "cage_elimination":
            return self.apply_cage_elimination()
        elif strategy_name == "complex_arithmetic_cages":
            return self.apply_complex_arithmetic_cages()
        elif strategy_name == "constraint_propagation":
            return self.apply_constraint_propagation()
        elif strategy_name == "multi_cage_analysis":
            return self.apply_multi_cage_analysis()
        
        # Hard strategies (simplified implementations)
        elif strategy_name == "advanced_cage_chaining":
            return self.apply_advanced_cage_chaining()
        elif strategy_name == "constraint_satisfaction_solving":
            return self.apply_constraint_satisfaction_solving()
        
        # If strategy not implemented, return False
        return False
    
    def apply_single_cell_cage(self) -> bool:
        """Apply single cell cage strategy"""
        progress = False
        for cage in self.cages:
            if len(cage['cells']) == 1:
                row, col = cage['cells'][0]
                if self.grid[row, col] == 0:
                    target_value = cage['target']
                    if 1 <= target_value <= self.grid_size:
                        self.grid[row, col] = target_value
                        self.update_candidates_after_assignment(row, col, target_value)
                        progress = True
        return progress
    
    def apply_simple_addition_cage(self) -> bool:
        """Apply simple addition cage strategy"""
        progress = False
        for cage in self.cages:
            if cage['operation'] == 'add' and len(cage['cells']) == 2:
                cells = cage['cells']
                target = cage['target']
                
                # Check if one cell is filled
                filled_cells = [(r, c) for r, c in cells if self.grid[r, c] != 0]
                empty_cells = [(r, c) for r, c in cells if self.grid[r, c] == 0]
                
                if len(filled_cells) == 1 and len(empty_cells) == 1:
                    filled_value = self.grid[filled_cells[0][0], filled_cells[0][1]]
                    required_value = target - filled_value
                    
                    if 1 <= required_value <= self.grid_size:
                        row, col = empty_cells[0]
                        self.grid[row, col] = required_value
                        self.update_candidates_after_assignment(row, col, required_value)
                        progress = True
        return progress
    
    def apply_simple_subtraction_cage(self) -> bool:
        """Apply simple subtraction cage strategy"""
        progress = False
        for cage in self.cages:
            if cage['operation'] == 'subtract' and len(cage['cells']) == 2:
                cells = cage['cells']
                target = cage['target']
                
                filled_cells = [(r, c) for r, c in cells if self.grid[r, c] != 0]
                empty_cells = [(r, c) for r, c in cells if self.grid[r, c] == 0]
                
                if len(filled_cells) == 1 and len(empty_cells) == 1:
                    filled_value = self.grid[filled_cells[0][0], filled_cells[0][1]]
                    
                    # Two possibilities: filled - x = target or x - filled = target
                    possible_values = [filled_value + target, filled_value - target]
                    valid_values = [v for v in possible_values if 1 <= v <= self.grid_size]
                    
                    if len(valid_values) == 1:
                        row, col = empty_cells[0]
                        self.grid[row, col] = valid_values[0]
                        self.update_candidates_after_assignment(row, col, valid_values[0])
                        progress = True
        return progress
    
    def apply_naked_single(self) -> bool:
        """Apply naked single strategy"""
        progress = False
        for (row, col), candidates in list(self.candidates.items()):
            if len(candidates) == 1:
                value = list(candidates)[0]
                self.grid[row, col] = value
                self.update_candidates_after_assignment(row, col, value)
                progress = True
        return progress
    
    def apply_hidden_single_row(self) -> bool:
        """Apply hidden single in row strategy"""
        progress = False
        for row in range(self.grid_size):
            for value in range(1, self.grid_size + 1):
                possible_cells = [(row, col) for col in range(self.grid_size) 
                                if (row, col) in self.candidates and value in self.candidates[(row, col)]]
                
                if len(possible_cells) == 1:
                    cell_row, cell_col = possible_cells[0]
                    self.grid[cell_row, cell_col] = value
                    self.update_candidates_after_assignment(cell_row, cell_col, value)
                    progress = True
        return progress
    
    def apply_hidden_single_column(self) -> bool:
        """Apply hidden single in column strategy"""
        progress = False
        for col in range(self.grid_size):
            for value in range(1, self.grid_size + 1):
                possible_cells = [(row, col) for row in range(self.grid_size) 
                                if (row, col) in self.candidates and value in self.candidates[(row, col)]]
                
                if len(possible_cells) == 1:
                    cell_row, cell_col = possible_cells[0]
                    self.grid[cell_row, cell_col] = value
                    self.update_candidates_after_assignment(cell_row, cell_col, value)
                    progress = True
        return progress
    
    def apply_cage_completion(self) -> bool:
        """Apply cage completion strategy"""
        progress = False
        for cage in self.cages:
            empty_cells = [(r, c) for r, c in cage['cells'] if self.grid[r, c] == 0]
            
            if len(empty_cells) == 1:
                row, col = empty_cells[0]
                assigned_values = [self.grid[r, c] for r, c in cage['cells'] if self.grid[r, c] != 0]
                
                possible_values = self.calculate_final_cage_value(cage, assigned_values)
                if len(possible_values) == 1:
                    value = list(possible_values)[0]
                    self.grid[row, col] = value
                    self.update_candidates_after_assignment(row, col, value)
                    progress = True
        return progress
    
    def apply_basic_multiplication_cage(self) -> bool:
        """Apply basic multiplication cage strategy"""
        progress = False
        for cage in self.cages:
            if cage['operation'] == 'multiply' and len(cage['cells']) == 2:
                cells = cage['cells']
                target = cage['target']
                
                filled_cells = [(r, c) for r, c in cells if self.grid[r, c] != 0]
                empty_cells = [(r, c) for r, c in cells if self.grid[r, c] == 0]
                
                if len(filled_cells) == 1 and len(empty_cells) == 1:
                    filled_value = self.grid[filled_cells[0][0], filled_cells[0][1]]
                    
                    if target % filled_value == 0:
                        required_value = target // filled_value
                        if 1 <= required_value <= self.grid_size:
                            row, col = empty_cells[0]
                            self.grid[row, col] = required_value
                            self.update_candidates_after_assignment(row, col, required_value)
                            progress = True
        return progress
    
    def apply_basic_division_cage(self) -> bool:
        """Apply basic division cage strategy"""
        progress = False
        for cage in self.cages:
            if cage['operation'] == 'divide' and len(cage['cells']) == 2:
                cells = cage['cells']
                target = cage['target']
                
                filled_cells = [(r, c) for r, c in cells if self.grid[r, c] != 0]
                empty_cells = [(r, c) for r, c in cells if self.grid[r, c] == 0]
                
                if len(filled_cells) == 1 and len(empty_cells) == 1:
                    filled_value = self.grid[filled_cells[0][0], filled_cells[0][1]]
                    
                    # Two possibilities: filled / x = target or x / filled = target
                    possible_values = []
                    if filled_value % target == 0:
                        possible_values.append(filled_value // target)
                    possible_values.append(filled_value * target)
                    
                    valid_values = [v for v in possible_values if 1 <= v <= self.grid_size]
                    
                    if len(valid_values) == 1:
                        row, col = empty_cells[0]
                        self.grid[row, col] = valid_values[0]
                        self.update_candidates_after_assignment(row, col, valid_values[0])
                        progress = True
        return progress
    
    # Moderate strategy implementations
    def apply_cage_elimination(self) -> bool:
        """Apply cage elimination strategy"""
        progress = False
        for cage in self.cages:
            for row, col in cage['cells']:
                if (row, col) in self.candidates:
                    old_candidates = self.candidates[(row, col)].copy()
                    new_candidates = self.get_cage_possible_values(cage, row, col)
                    self.candidates[(row, col)] = self.candidates[(row, col)].intersection(new_candidates)
                    
                    if len(self.candidates[(row, col)]) < len(old_candidates):
                        progress = True
        return progress
    
    def apply_complex_arithmetic_cages(self) -> bool:
        """Apply complex arithmetic cages strategy"""
        # Simplified implementation - use cage elimination for larger cages
        return self.apply_cage_elimination()
    
    def apply_constraint_propagation(self) -> bool:
        """Apply constraint propagation strategy"""
        # Simplified implementation - propagate all constraints
        progress = False
        progress |= self.apply_cage_elimination()
        progress |= self.apply_naked_single()
        progress |= self.apply_hidden_single_row()
        progress |= self.apply_hidden_single_column()
        return progress
    
    def apply_multi_cage_analysis(self) -> bool:
        """Apply multi-cage analysis strategy"""
        # Simplified implementation - analyze overlapping cages
        return False  # Complex implementation needed
    
    # Hard strategy implementations (simplified)
    def apply_advanced_cage_chaining(self) -> bool:
        """Apply advanced cage chaining strategy"""
        # Simplified implementation
        return self.apply_constraint_propagation()
    
    def apply_constraint_satisfaction_solving(self) -> bool:
        """Apply constraint satisfaction solving strategy"""
        # Simplified implementation - use backtracking for small problems
        return False  # Complex implementation needed
    
    def validate_solution(self) -> bool:
        """Validate if the current grid state is a valid solution"""
        # Check if all cells are filled
        if np.any(self.grid == 0):
            return False
        
        # Check Latin square property
        for i in range(self.grid_size):
            # Check row
            if set(self.grid[i, :]) != set(range(1, self.grid_size + 1)):
                return False
            # Check column
            if set(self.grid[:, i]) != set(range(1, self.grid_size + 1)):
                return False
        
        # Check cage constraints
        for cage in self.cages:
            cage_values = [self.grid[r, c] for r, c in cage['cells']]
            if not self.evaluate_cage_constraint(cage_values, cage['operation'], cage['target']):
                return False
        
        return True
    
    def get_strategy_application_order(self, difficulty: str) -> List[str]:
        """Get the recommended order for applying strategies based on difficulty"""
        if difficulty == 'easy':
            return [
                'naked_single',
                'single_cell_cage',
                'cage_completion',
                'simple_addition_cage',
                'simple_subtraction_cage',
                'basic_multiplication_cage',
                'basic_division_cage',
                'hidden_single_row',
                'hidden_single_column'
            ]
        elif difficulty == 'moderate':
            return [
                'naked_single',
                'single_cell_cage',
                'cage_completion',
                'simple_addition_cage',
                'cage_elimination',
                'constraint_propagation',
                'complex_arithmetic_cages',
                'hidden_single_row',
                'hidden_single_column'
            ]
        else:  # hard
            return [
                'naked_single',
                'single_cell_cage',
                'cage_completion',
                'cage_elimination',
                'constraint_propagation',
                'advanced_cage_chaining',
                'constraint_satisfaction_solving',
                'complex_arithmetic_cages'
            ]