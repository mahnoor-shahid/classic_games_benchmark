# 5x5 Sudoku Validator (digits 0-4)
import numpy as np
from typing import List, Set, Tuple, Optional

class SudokuValidator:
    def __init__(self):
        self.valid_digits = set(range(5))

    def is_valid_puzzle(self, puzzle: np.ndarray) -> bool:
        """Check if a puzzle is valid (no duplicate numbers in rows, columns, or diagonals)"""
        # Check rows
        for row in range(5):
            row_values = puzzle[row, :]
            if not self._is_valid_unit(row_values):
                return False
        # Check columns
        for col in range(5):
            col_values = puzzle[:, col]
            if not self._is_valid_unit(col_values):
                return False
        # Check main diagonal
        diag_values = np.array([puzzle[i, i] for i in range(5)])
        if not self._is_valid_unit(diag_values):
            return False
        # Check anti-diagonal
        anti_diag_values = np.array([puzzle[i, 4 - i] for i in range(5)])
        if not self._is_valid_unit(anti_diag_values):
            return False
        return True

    def _is_valid_unit(self, values: np.ndarray) -> bool:
        filled_values = values[values != -1]
        if len(filled_values) != len(set(filled_values)):
            return False
        if not all(0 <= x <= 4 for x in filled_values):
            return False
        return True

    def has_unique_solution(self, puzzle: np.ndarray) -> bool:
        """Check if a puzzle has a unique solution (brute-force for 5x5)"""
        solutions_found = 0
        max_solutions = 2
        def solve(grid):
            nonlocal solutions_found
            if solutions_found >= max_solutions:
                return
            empty = self._find_empty_cell(grid)
            if not empty:
                solutions_found += 1
                return
            row, col = empty
            for num in range(5):
                if self._is_valid_placement(grid, row, col, num):
                    grid[row, col] = num
                    solve(grid)
                    grid[row, col] = -1
        test_grid = puzzle.copy()
        solve(test_grid)
        return solutions_found == 1

    def _find_empty_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        for i in range(5):
            for j in range(5):
                if grid[i, j] == -1:
                    return (i, j)
        return None

    def _is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        if row == col and num in [grid[i, i] for i in range(5) if i != row]:
            return False
        if row + col == 4 and num in [grid[i, 4 - i] for i in range(5) if i != row]:
            return False
        return True

    def requires_strategies(self, puzzle: np.ndarray, strategies: List[str]) -> bool:
        """Check if the puzzle requires the specified strategies"""
        # For now, we'll use a simple heuristic based on filled cells
        filled_cells = np.sum(puzzle != -1)

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
            return np.sum(region != -1)

        if symmetry_type == 'row':
            # Check if top and bottom halves have similar number of filled cells
            top_half = count_filled_region(puzzle[:2])
            bottom_half = count_filled_region(puzzle[3:])
            return abs(top_half - bottom_half) <= 2

        elif symmetry_type == 'box':
            # Check if left and right sides have similar number of filled cells
            left_side = count_filled_region(puzzle[:, :2])
            right_side = count_filled_region(puzzle[:, 3:])
            return abs(left_side - right_side) <= 2

        elif symmetry_type == 'cross':
            # Check if diagonals have similar number of filled cells
            diag1 = count_filled_region(np.diag(puzzle))
            diag2 = count_filled_region(np.diag(np.fliplr(puzzle)))
            return abs(diag1 - diag2) <= 2

        elif symmetry_type == 'diamond':
            # Check if all quadrants have similar number of filled cells
            q1 = count_filled_region(puzzle[:2, :2])
            q2 = count_filled_region(puzzle[:2, 3:])
            q3 = count_filled_region(puzzle[3:, :2])
            q4 = count_filled_region(puzzle[3:, 3:])
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
        actual_filled = np.sum(puzzle != -1)
        if abs(actual_filled - filled_cells) > 2:
            return False

        # Check if filled cells are reasonably distributed
        for i in range(5):
            # Check row distribution
            row_filled = np.sum(puzzle[i, :] != -1)
            if row_filled < 1:  # Allow at least one filled cell per row
                return False
            # Check column distribution
            col_filled = np.sum(puzzle[:, i] != -1)
            if col_filled < 1:  # Allow at least one filled cell per column
                return False

        return True 