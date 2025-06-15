import numpy as np
from typing import List, Set, Tuple, Optional

class SudokuValidator:
    def __init__(self):
        self.valid_digits = set(range(1, 10))
        
    def is_valid_puzzle(self, puzzle: np.ndarray) -> bool:
        """Check if a puzzle is valid (no duplicate numbers in rows, columns, or boxes)"""
        # Check rows
        for row in range(9):
            row_values = puzzle[row, :]
            if not self._is_valid_unit(row_values):
                return False
                
        # Check columns
        for col in range(9):
            col_values = puzzle[:, col]
            if not self._is_valid_unit(col_values):
                return False
                
        # Check boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box_values = puzzle[box_row:box_row+3, box_col:box_col+3].flatten()
                if not self._is_valid_unit(box_values):
                    return False
                    
        return True
        
    def _is_valid_unit(self, values: np.ndarray) -> bool:
        """Check if a unit (row, column, or box) is valid"""
        # Remove zeros (empty cells)
        filled_values = values[values != 0]
        
        # Check if any number appears more than once
        if len(filled_values) != len(set(filled_values)):
            return False
            
        # Check if all numbers are valid
        if not all(1 <= x <= 9 for x in filled_values):
            return False
            
        return True
        
    def has_unique_solution(self, puzzle: np.ndarray) -> bool:
        """Check if a puzzle has a unique solution"""
        # Make a copy of the puzzle
        puzzle_copy = puzzle.copy()
        
        # Try to solve the puzzle
        if not self._solve_sudoku(puzzle_copy):
            return False
            
        # Try to find another solution
        for row in range(9):
            for col in range(9):
                if puzzle[row, col] == 0:
                    # Try a different number in this cell
                    for num in range(1, 10):
                        if num != puzzle_copy[row, col]:
                            puzzle_copy = puzzle.copy()
                            puzzle_copy[row, col] = num
                            if self._solve_sudoku(puzzle_copy):
                                return False
                                
        return True
        
    def _solve_sudoku(self, grid: np.ndarray) -> bool:
        """Solve sudoku using backtracking"""
        empty_cell = self._find_empty_cell(grid)
        if not empty_cell:
            return True
            
        row, col = empty_cell
        for num in range(1, 10):
            if self._is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                if self._solve_sudoku(grid):
                    return True
                grid[row, col] = 0
                
        return False
        
    def _find_empty_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find an empty cell in the grid"""
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return (i, j)
        return None
        
    def _is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if a number can be placed in a cell"""
        # Check row
        if num in grid[row, :]:
            return False
            
        # Check column
        if num in grid[:, col]:
            return False
            
        # Check box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
            
        return True
        
    def requires_strategies(self, puzzle: np.ndarray, strategies: List[str]) -> bool:
        """Check if the puzzle requires the specified strategies"""
        # For now, we'll use a simple heuristic based on filled cells
        filled_cells = np.sum(puzzle != 0)
        
        # Basic strategies are always required
        basic_strategies = {'naked_single', 'hidden_single_row', 'hidden_single_column', 'hidden_single_box'}
        if not any(s in basic_strategies for s in strategies):
            return False
            
        # Check if puzzle difficulty matches strategy complexity
        if filled_cells <= 25:  # Hard puzzle
            if not any(s in {'x_wing', 'swordfish', 'xy_wing'} for s in strategies):
                return False
        elif filled_cells <= 30:  # Moderate puzzle
            if not any(s in {'naked_pair', 'hidden_pair', 'pointing_pair'} for s in strategies):
                return False
                
        return True
        
    def validate_compositionality(self, puzzle: np.ndarray, template: dict) -> bool:
        """Validate that the puzzle maintains proper compositionality with its strategies"""
        # Check symmetry
        if not self._validate_symmetry(puzzle, template['symmetry_type']):
            return False
            
        # Check strategy sequence
        if not self._validate_strategy_sequence(puzzle, template['required_strategies']):
            return False
            
        # Check cell relationships
        if not self._validate_cell_relationships(puzzle, template['filled_cells']):
            return False
            
        return True
        
    def _validate_symmetry(self, puzzle: np.ndarray, symmetry_type: str) -> bool:
        """Check if the puzzle adheres to the required symmetry"""
        # Count filled cells in each region
        def count_filled_region(region):
            return np.sum(region != 0)
            
        if symmetry_type == 'row':
            # Check if top and bottom halves have similar number of filled cells
            top_half = count_filled_region(puzzle[:4])
            bottom_half = count_filled_region(puzzle[5:])
            return abs(top_half - bottom_half) <= 2
            
        elif symmetry_type == 'box':
            # Check if left and right sides have similar number of filled cells
            left_side = count_filled_region(puzzle[:, :3])
            right_side = count_filled_region(puzzle[:, 6:])
            return abs(left_side - right_side) <= 2
            
        elif symmetry_type == 'cross':
            # Check if diagonals have similar number of filled cells
            diag1 = count_filled_region(np.diag(puzzle))
            diag2 = count_filled_region(np.diag(np.fliplr(puzzle)))
            return abs(diag1 - diag2) <= 2
            
        elif symmetry_type == 'diamond':
            # Check if all quadrants have similar number of filled cells
            q1 = count_filled_region(puzzle[:4, :4])
            q2 = count_filled_region(puzzle[:4, 5:])
            q3 = count_filled_region(puzzle[5:, :4])
            q4 = count_filled_region(puzzle[5:, 5:])
            return max(abs(q1 - q2), abs(q1 - q3), abs(q1 - q4)) <= 3
            
        elif symmetry_type == 'diagonal':
            # Check if main diagonal has similar number of filled cells as anti-diagonal
            main_diag = count_filled_region(np.diag(puzzle))
            anti_diag = count_filled_region(np.diag(np.fliplr(puzzle)))
            return abs(main_diag - anti_diag) <= 2
            
        return True
        
    def _validate_strategy_sequence(self, puzzle: np.ndarray, required_strategies: List[str]) -> bool:
        """Ensure that basic strategies are present when advanced strategies are used"""
        # Basic strategies that should always be present
        basic_strategies = {'naked_single', 'hidden_single_row', 'hidden_single_column', 'hidden_single_box'}
        
        # Check if at least one basic strategy is present
        if not any(s in required_strategies for s in basic_strategies):
            return False
            
        # Check if advanced strategies have their prerequisites
        if 'x_wing' in required_strategies and 'naked_pair' not in required_strategies:
            return False
        if 'swordfish' in required_strategies and 'x_wing' not in required_strategies:
            return False
            
        return True
        
    def _validate_cell_relationships(self, puzzle: np.ndarray, filled_cells: int) -> bool:
        """Check that filled cells maintain proper relationships"""
        # Check if number of filled cells is within acceptable range
        actual_filled = np.sum(puzzle != 0)
        if abs(actual_filled - filled_cells) > 2:
            return False
            
        # Check if filled cells are reasonably distributed
        for i in range(9):
            # Check row distribution
            row_filled = np.sum(puzzle[i, :] != 0)
            if row_filled < 1:  # Allow at least one filled cell per row
                return False
            # Check column distribution
            col_filled = np.sum(puzzle[:, i] != 0)
            if col_filled < 1:  # Allow at least one filled cell per column
                return False
                
        return True 